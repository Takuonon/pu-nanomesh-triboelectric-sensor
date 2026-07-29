#include <Arduino.h>
#include <ArduinoBLE.h>
#include <PDM.h>

extern "C" {
  #include "nrfx_ppi.h"
  #include "nrf_saadc.h"
  #include "nrf_timer.h"
}

// ===================== BLE UUID =====================
BLEService audioService("19B20000-E8F2-537E-4F6C-D104768A1214");

BLEByteCharacteristic ctrlChar(
  "19B20001-E8F2-537E-4F6C-D104768A1214",
  BLERead | BLEWrite
);

BLECharacteristic audioChar(
  "19B20002-E8F2-537E-4F6C-D104768A1214",
  BLEIndicate,
  244
);

// ===================== Settings =====================
static const int PDM_CAPTURE_RATE = 16000;
static const int ADC_CAPTURE_RATE = 8000;
static const int STORE_RATE = 8000;

static const int MAX_NUM_SAMPLES = 32000;
static const int MAX_REC_SECONDS = MAX_NUM_SAMPLES / STORE_RATE; // 4

static constexpr uint32_t CHUNK_SAMPLES = 1024;
static const float WARMUP_DROP_SECONDS = 1.0f;

static const int PAYLOAD_BYTES = 200;
static const int FRAMES_PER_PKT = PAYLOAD_BYTES / 4;

// Shutdown command uses a value outside the normal capture-duration range.
static const byte CMD_SHUTDOWN = 0xFF; // 255

// ===================== Buffers =====================
static int16_t ch1_adc[MAX_NUM_SAMPLES];
static int16_t ch2_pdm[MAX_NUM_SAMPLES];

static volatile int adcWritten = 0;
static volatile int pdmWritten = 0;

static int16_t pdmBuf[256];

static int16_t saadcBuf0[CHUNK_SAMPLES];
static int16_t saadcBuf1[CHUNK_SAMPLES];
static int16_t* activeBuf = saadcBuf0;
static int16_t* nextBuf   = saadcBuf1;

// ===================== Recording control =====================
static volatile bool recording = false;
static volatile int targetSamples = STORE_RATE;
static volatile int warmupDropPDM = 0;
static volatile int warmupDropADC = 0;

static volatile uint8_t pdmDecimPhase = 0;

// ===================== TIMER4 + PPI =====================
static NRF_TIMER_Type* const TMR = NRF_TIMER4;
static nrf_ppi_channel_t ppiCh;

// ===================== LED =====================
static const bool LED_ACTIVE_LOW = true;

enum DeviceState {
  STATE_IDLE_DISCONNECTED = 0,
  STATE_IDLE_CONNECTED = 1,
  STATE_RECORDING = 2,
  STATE_COMMUNICATING = 3,
};

static DeviceState gState = STATE_IDLE_DISCONNECTED;
static unsigned long gBlinkLastMs = 0;
static bool gBlinkOn = false;

static inline void ledWrite(uint8_t pin, bool on) {
  if (LED_ACTIVE_LOW) digitalWrite(pin, on ? LOW : HIGH);
  else digitalWrite(pin, on ? HIGH : LOW);
}
static inline void ledsAllOff() {
  ledWrite(LEDR, false);
  ledWrite(LEDG, false);
  ledWrite(LEDB, false);
}
static void setState(DeviceState st) {
  gState = st;
  if (gState == STATE_RECORDING) {
    ledWrite(LEDR, false); ledWrite(LEDG, false); ledWrite(LEDB, true);   // blue
  } else if (gState == STATE_COMMUNICATING) {
    ledWrite(LEDR, false); ledWrite(LEDG, true);  ledWrite(LEDB, false);  // green
  } else {
    ledsAllOff();
    gBlinkLastMs = millis();
    gBlinkOn = false;
  }
}
static void updateIdleBlink() {
  if (gState != STATE_IDLE_DISCONNECTED && gState != STATE_IDLE_CONNECTED) return;
  const unsigned long now = millis();
  if (now - gBlinkLastMs >= 500) {
    gBlinkLastMs = now;
    gBlinkOn = !gBlinkOn;
    if (gState == STATE_IDLE_DISCONNECTED) {
      ledWrite(LEDR, gBlinkOn); ledWrite(LEDG, false); ledWrite(LEDB, false);
    } else {
      ledWrite(LEDR, false); ledWrite(LEDG, false); ledWrite(LEDB, gBlinkOn);
    }
  }
}

// ===================== PDM callback =====================
void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable <= 0) return;

  int samples = bytesAvailable / 2;
  if (samples > (int)(sizeof(pdmBuf) / sizeof(pdmBuf[0]))) {
    samples = sizeof(pdmBuf) / sizeof(pdmBuf[0]);
  }
  PDM.read(pdmBuf, samples * 2);

  if (!recording) return;

  for (int i = 0; i < samples; i++) {
    if (warmupDropPDM > 0) {
      warmupDropPDM--;
      if (warmupDropPDM == 0) {
        pdmDecimPhase = 0;
      }
      continue;
    }

    if (pdmDecimPhase) {
      pdmDecimPhase = 0;
      continue;
    }
    pdmDecimPhase = 1;

    int idx = pdmWritten;
    if (idx >= targetSamples) {
      return;
    }
    ch2_pdm[idx] = pdmBuf[i];
    pdmWritten++;
  }
}

// ===================== SAADC init (register) =====================
static void timer4_init_16khz() {
  TMR->TASKS_STOP  = 1;
  TMR->TASKS_CLEAR = 1;
  TMR->MODE      = TIMER_MODE_MODE_Timer;
  TMR->BITMODE   = TIMER_BITMODE_BITMODE_32Bit;
  TMR->PRESCALER = 0;
  TMR->CC[0]     = 16000000UL / ADC_CAPTURE_RATE;
  TMR->SHORTS    = TIMER_SHORTS_COMPARE0_CLEAR_Msk;
  TMR->EVENTS_COMPARE[0] = 0;
}

static void saadc_init_ain0_reg() {
  NRF_SAADC->ENABLE = (SAADC_ENABLE_ENABLE_Enabled << SAADC_ENABLE_ENABLE_Pos);
  NRF_SAADC->RESOLUTION = SAADC_RESOLUTION_VAL_12bit;

  NRF_SAADC->CH[0].PSELP = SAADC_CH_PSELP_PSELP_AnalogInput0;
  NRF_SAADC->CH[0].PSELN = SAADC_CH_PSELN_PSELN_NC;

  NRF_SAADC->CH[0].CONFIG =
      (SAADC_CH_CONFIG_REFSEL_Internal << SAADC_CH_CONFIG_REFSEL_Pos) |
      (SAADC_CH_CONFIG_GAIN_Gain1_6     << SAADC_CH_CONFIG_GAIN_Pos)   |
      (SAADC_CH_CONFIG_TACQ_10us        << SAADC_CH_CONFIG_TACQ_Pos)   |
      (SAADC_CH_CONFIG_MODE_SE          << SAADC_CH_CONFIG_MODE_Pos)   |
      (SAADC_CH_CONFIG_RESP_Bypass      << SAADC_CH_CONFIG_RESP_Pos)   |
      (SAADC_CH_CONFIG_RESN_Bypass      << SAADC_CH_CONFIG_RESN_Pos);

  activeBuf = saadcBuf0;
  nextBuf   = saadcBuf1;

  NRF_SAADC->RESULT.PTR    = (uint32_t)activeBuf;
  NRF_SAADC->RESULT.MAXCNT = CHUNK_SAMPLES;

  NRF_SAADC->EVENTS_END = 0;
  NRF_SAADC->EVENTS_STARTED = 0;
}

static void ppi_connect_timer_to_saadc_sample() {
  nrfx_ppi_channel_alloc(&ppiCh);
  const uint32_t evt_addr  = (uint32_t)&TMR->EVENTS_COMPARE[0];
  const uint32_t task_addr = (uint32_t)&NRF_SAADC->TASKS_SAMPLE;
  nrfx_ppi_channel_assign(ppiCh, evt_addr, task_addr);
  nrfx_ppi_channel_enable(ppiCh);
}

static inline void consume_saadc_samples(const int16_t* buf, int n) {
  for (int i = 0; i < n; i++) {
    if (warmupDropADC > 0) {
      warmupDropADC--;
      continue;
    }
    int idx = adcWritten;
    if (idx >= targetSamples) return;
    ch1_adc[idx] = buf[i];
    adcWritten++;
  }
}

// ===================== Record control =====================
static bool startRecordingBoth(int seconds) {
  if (seconds < 1) seconds = 1;
  if (seconds > MAX_REC_SECONDS) seconds = MAX_REC_SECONDS;

  targetSamples = STORE_RATE * seconds;

  adcWritten = 0;
  pdmWritten = 0;

  warmupDropPDM = (int)(PDM_CAPTURE_RATE * WARMUP_DROP_SECONDS);
  warmupDropADC = (int)(ADC_CAPTURE_RATE * WARMUP_DROP_SECONDS);
  pdmDecimPhase = 0;

  recording = true;

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, PDM_CAPTURE_RATE)) {
    recording = false;
    return false;
  }

  NRF_SAADC->EVENTS_CALIBRATEDONE = 0;
  NRF_SAADC->TASKS_CALIBRATEOFFSET = 1;
  delay(10);

  NRF_SAADC->EVENTS_STARTED = 0;
  NRF_SAADC->TASKS_START = 1;
  for (volatile int i = 0; i < 20000 && NRF_SAADC->EVENTS_STARTED == 0; i++) {}

  NRF_SAADC->TASKS_SAMPLE = 1;

  TMR->TASKS_CLEAR = 1;
  TMR->TASKS_START = 1;

  return true;
}

static void stopRecordingBoth() {
  recording = false;

  TMR->TASKS_STOP = 1;
  NRF_SAADC->TASKS_STOP = 1;
  PDM.end();
}

// ===================== Low-power shutdown =====================
// Enter this path when CMD_SHUTDOWN (255) is received over BLE.
// Wake-up from System OFF requires pressing RESET or power-cycling the board.
static void enterSystemOffNow() {
  // Show shutdown acknowledgement in purple (red + blue).
  ledWrite(LEDR, true);
  ledWrite(LEDG, false);
  ledWrite(LEDB, true);
  delay(150);  // Briefly visible to the user.

  // Turn LEDs off just before shutdown.
  ledsAllOff();
  delay(50);

  // Stop peripherals defensively.
  stopRecordingBoth();
  nrfx_ppi_channel_disable(ppiCh);
  NRF_SAADC->ENABLE = (SAADC_ENABLE_ENABLE_Disabled << SAADC_ENABLE_ENABLE_Pos);

  // Give BLE a short grace period to stop cleanly.
  BLE.stopAdvertise();
  BLE.disconnect();
  delay(100);

  // Enter System OFF.
  NRF_POWER->SYSTEMOFF = 1;

  // Execution should not return here.
  while (1) {}
}

// ===================== Send: stereo interleave =====================
static void sendRecordedStereoInterleaved(int nSamples) {
  uint16_t seq = 0;
  uint8_t packet[2 + FRAMES_PER_PKT * 4];

  int idx = 0;
  while (idx < nSamples) {
    int frames = nSamples - idx;
    if (frames > FRAMES_PER_PKT) frames = FRAMES_PER_PKT;

    packet[0] = (uint8_t)(seq & 0xFF);
    packet[1] = (uint8_t)((seq >> 8) & 0xFF);

    uint8_t* p = packet + 2;
    for (int i = 0; i < frames; i++) {
      int16_t a = ch1_adc[idx + i];
      int16_t b = ch2_pdm[idx + i];

      *p++ = (uint8_t)(a & 0xFF);
      *p++ = (uint8_t)((a >> 8) & 0xFF);
      *p++ = (uint8_t)(b & 0xFF);
      *p++ = (uint8_t)((b >> 8) & 0xFF);
    }

    audioChar.writeValue(packet, 2 + frames * 4);
    idx += frames;
    seq++;
    delay(5);
  }
}

// ===================== BLE main =====================
void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 2000) delay(10);

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  ledsAllOff();
  setState(STATE_IDLE_DISCONNECTED);

  timer4_init_16khz();
  saadc_init_ain0_reg();
  ppi_connect_timer_to_saadc_sample();

  if (!BLE.begin()) {
    Serial.println("BLE.begin failed");
    while (1) {}
  }

  BLE.setLocalName("XIAO-AUDIO");
  BLE.setDeviceName("XIAO-AUDIO");
  BLE.setAdvertisedService(audioService);

  audioService.addCharacteristic(ctrlChar);
  audioService.addCharacteristic(audioChar);
  BLE.addService(audioService);

  ctrlChar.writeValue((byte)0);

  BLE.advertise();
  Serial.println("BLE Audio Peripheral Ready (ADC ch1 + PDM ch2, interleaved)");
}

void loop() {
  BLEDevice central = BLE.central();
  if (!central) {
    if (gState != STATE_IDLE_DISCONNECTED) setState(STATE_IDLE_DISCONNECTED);
    updateIdleBlink();
    delay(10);
    return;
  }

  Serial.print("Connected: ");
  Serial.println(central.address());
  setState(STATE_IDLE_CONNECTED);

  while (central.connected()) {
    updateIdleBlink();

    if (ctrlChar.written()) {
      byte cmd = ctrlChar.value();

      // Shutdown command (255).
      if (cmd == CMD_SHUTDOWN) {
        Serial.println("CMD_SHUTDOWN received. Going SYSTEM OFF...");
        // Optionally write an acknowledgement to CTRL here if needed.
        // ctrlChar.writeValue((byte)0);
        delay(50);
        enterSystemOffNow();
      }

      // Normal commands: 1..MAX_REC_SECONDS represent capture duration in seconds.
      if (cmd >= 1 && cmd <= (byte)MAX_REC_SECONDS) {
        byte secs = cmd;

        setState(STATE_RECORDING);

        bool ok = startRecordingBoth((int)secs);
        if (!ok) {
          setState(STATE_IDLE_CONNECTED);
          ctrlChar.writeValue((byte)0);
          continue;
        }

        while (central.connected()) {
          if (NRF_SAADC->EVENTS_END) {
            NRF_SAADC->EVENTS_END = 0;

            TMR->TASKS_STOP = 1;

            consume_saadc_samples(activeBuf, (int)CHUNK_SAMPLES);

            if (adcWritten >= targetSamples && pdmWritten >= targetSamples) {
              break;
            }

            int16_t* tmp = activeBuf;
            activeBuf = nextBuf;
            nextBuf = tmp;

            NRF_SAADC->RESULT.PTR    = (uint32_t)activeBuf;
            NRF_SAADC->RESULT.MAXCNT = CHUNK_SAMPLES;

            NRF_SAADC->TASKS_STOP = 1;
            NRF_SAADC->TASKS_START = 1;
            NRF_SAADC->TASKS_SAMPLE = 1;

            TMR->TASKS_CLEAR = 1;
            TMR->TASKS_START = 1;
          }

          delay(1);
        }

        stopRecordingBoth();

        int nSend = adcWritten;
        if (pdmWritten < nSend) nSend = pdmWritten;
        if (nSend > targetSamples) nSend = targetSamples;

        Serial.print("Recorded adc=");
        Serial.print(adcWritten);
        Serial.print(" pdm=");
        Serial.print(pdmWritten);
        Serial.print(" -> send=");
        Serial.println(nSend);

        setState(STATE_COMMUNICATING);
        sendRecordedStereoInterleaved(nSend);
        setState(STATE_IDLE_CONNECTED);

        ctrlChar.writeValue((byte)0);
        Serial.println("Send done.");
      }
    }

    delay(5);
  }

  Serial.print("Disconnected: ");
  Serial.println(central.address());
  setState(STATE_IDLE_DISCONNECTED);
}
