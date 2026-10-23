# DriveGuard-X

## Context-Aware Adaptive Battery Protection System for Electric Vehicles
### ESP32-powered 4WD robot-car prototype · Battery + IMU + Audio sensing · FastAPI inference backend · MATLAB/Simulink simulation · Cause-aware adaptive protection

---

## How to read this document

This README is the single source of truth for the DriveGuard-X project. It combines:

1. the **research concept** (what is being investigated and why),
2. the **AI/ML design** (three models, fusion, attribution, protection),
3. the **software stack** (FastAPI, MATLAB/Simulink, dashboard),
4. the **physical prototype** (4WD ESP32 robot car, battery pack, power, wiring),
5. the **data, experiment, and evaluation plan**,
6. the **safety and verification checklist**, and
7. the **open items** that are not yet confirmed.

### Status legend

Every hardware or design statement should be read with its status in mind.

| Tag | Meaning |
|---|---|
| **[CONFIRMED]** | Decided, or confirmed by the project owner |
| **[PLANNED]** | Intended design; not yet built or tested |
| **[PROPOSED]** | Suggested starting value or approach; must be validated |
| **[VERIFY]** | Must be checked against the exact part, label, or datasheet before use |
| **[OPEN]** | Unresolved; listed in the Open Items Register (Section 25) |

Nothing in this document should be treated as physically tested unless it is explicitly recorded as tested in the project log. Numeric values given as examples (PWM limits, thresholds, window sizes, sampling rates) are **starting points, not results**.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem and Motivation](#2-problem-and-motivation)
3. [Research Question, Hypotheses, and Novelty](#3-research-question-hypotheses-and-novelty)
4. [System Architecture](#4-system-architecture)
5. [Sensing Modalities (Final Design Decision)](#5-sensing-modalities-final-design-decision)
6. [Physical Prototype: 4WD Robot Car](#6-physical-prototype-4wd-robot-car)
7. [Hardware Bill of Materials and Status](#7-hardware-bill-of-materials-and-status)
8. [Battery Pack, BMS, and Charging](#8-battery-pack-bms-and-charging)
9. [Power Distribution](#9-power-distribution)
10. [Motor Control](#10-motor-control)
11. [ESP32 Board and Pin Map](#11-esp32-board-and-pin-map)
12. [Component-Level Wiring](#12-component-level-wiring)
13. [Sensor Details and Calibration](#13-sensor-details-and-calibration)
14. [ESP32 Firmware Design](#14-esp32-firmware-design)
15. [Communication Protocol](#15-communication-protocol)
16. [AI Models](#16-ai-models)
17. [Fusion, Attribution, and Adaptive Protection](#17-fusion-attribution-and-adaptive-protection)
18. [FastAPI Backend](#18-fastapi-backend)
19. [MATLAB / Simulink Simulation](#19-matlab--simulink-simulation)
20. [Dashboard](#20-dashboard)
21. [Data Collection, Labeling, and Leakage Control](#21-data-collection-labeling-and-leakage-control)
22. [Domain Shift, Adaptation, and Augmentation](#22-domain-shift-adaptation-and-augmentation)
23. [Experiments and Evaluation](#23-experiments-and-evaluation)
24. [Safety and Validation Procedure](#24-safety-and-validation-procedure)
25. [Open Items Register](#25-open-items-register)
26. [Development Roadmap and Review Milestones](#26-development-roadmap-and-review-milestones)
27. [Repository Structure](#27-repository-structure)
28. [Technology Stack](#28-technology-stack)
29. [Risks and Mitigations](#29-risks-and-mitigations)
30. [Working Rules for Assistants and Collaborators](#30-working-rules-for-assistants-and-collaborators)
31. [Final Definition of DriveGuard-X](#31-final-definition-of-driveguard-x)

---

# 1. Executive Summary

DriveGuard-X is a hardware-software integrated battery protection system for electric vehicles, prototyped on an ESP32-controlled four-wheel robot car.

Conventional Battery Management Systems decide mainly from battery-side measurements (voltage, current, temperature, SOC, SOH). That is effective for direct electrical and thermal faults, but a high-current or high-stress event does not reveal *why* it happened. The same current spike can be caused by:

- the **driver** (aggressive acceleration),
- the **road** (rough surface, bump), or
- the **battery** (degraded health).

DriveGuard-X combines three independent sensing modalities, each with its own sensor and its own AI model:

| Modality | Sensor(s) | AI model | Question answered |
|---|---|---|---|
| Battery | Voltage sensor, ACS712, DS18B20 | Battery model (XGBoost / Random Forest) | What is the battery's state and stress level? |
| Driving behavior | MPU6050 (accelerometer + gyroscope) | IMU model | How is the vehicle being driven? |
| Road condition | INMP441 MEMS microphone (**audio only**) | Audio road model | What is the road surface like? |

The three outputs are fused, a rule-based attribution engine labels the likely contributing cause of a stress event (driver / road / battery / mixed) with a confidence, and an adaptive protection layer selects a bounded motor/PWM response. The result is compared experimentally against a conventional current-threshold baseline.

The project is built in two stages:

- **Stage 1 (software and simulation):** public datasets → three models → FastAPI → MATLAB/Simulink closed loop → dashboard.
- **Stage 2 (hardware and validation):** build the 4WD robot car → collect labeled RC data → adapt models to the platform → real-time closed-loop protection → baseline comparison.

> **Central claim to test:** context-aware cause attribution enables more appropriate battery protection than a threshold-only strategy.

---

# 2. Problem and Motivation

## 2.1 Conventional protection

```text
Battery Current
      |
      v
Compare with Threshold
      |
      +---- Normal ----> Continue
      |
      +---- High   ----> Reduce Power
```

Simple and useful, but blind to context.

## 2.2 Three situations that look alike electrically

### Situation A: Aggressive acceleration (driver-caused)

```text
Aggressive Throttle -> High Motor Torque -> High Battery Current
```
The battery may be healthy. The current is driver demand.

### Situation B: Rough road (road-caused)

```text
Rough Road -> Vibration / Changing Load -> Current Fluctuation
```
The battery may be healthy. The event is environmental.

### Situation C: Degraded battery (battery-caused)

```text
Normal Driver + Smooth Road + Low SOH -> Battery Stress
```
Strong protection matters most here.

A threshold-only system sees similar electrical symptoms in all three. DriveGuard-X attempts to tell them apart and respond proportionately.

## 2.3 Consequences of not distinguishing causes

- **Unnecessary power cuts** (false interventions) when a healthy battery meets a transient driver or road event, which degrades vehicle performance.
- **Inadequate protection** (missed events) when a genuinely degraded battery is treated like a transient.

---

# 3. Research Question, Hypotheses, and Novelty

## 3.1 Core research question

> **Can contextual information from battery measurements, driving behavior, and road conditions be used to identify the likely cause of battery-stress events and enable more appropriate adaptive protection compared with conventional threshold-based protection?**

## 3.2 Working hypotheses

| ID | Hypothesis |
|---|---|
| H1 | A battery model trained on public data, then target-domain calibrated, can estimate battery state/stress usefully on the RC pack. |
| H2 | An IMU model can classify driving behavior on the RC platform and remain robust to road vibration. |
| H3 | An audio-only model can classify road condition on the RC platform when trained with public data and adapted with RC recordings. |
| H4 | Fusing the three outputs allows more accurate cause attribution than any single modality. |
| H5 | Cause-aware protection produces fewer false interventions and fewer missed events than threshold-only protection, at comparable or better battery stress and vehicle performance. |

These are hypotheses to be tested, not expected results.

## 3.3 Attribution is inference, not proof of causation

The attribution engine reports the **likely contributing cause** given the available evidence. Correlation between a stress event and a modality's state does not prove causation. The system must:

- present attribution with a confidence value,
- expose the evidence from each modality,
- distinguish **predictions** (model outputs) from **measured facts** (sensor readings),
- reduce confidence when evidence is missing or contradictory.

## 3.4 Novelty positioning

The following are **not** sufficient novelty claims by themselves, because substantial prior work exists:

- AI-based battery SOH/RUL estimation
- XGBoost battery prediction
- IMU-based driving behavior classification
- Microphone/tire-road acoustic road classification
- ESP32-based BMS
- Sensor fusion in general
- FastAPI as an inference backend

The contribution is the **system-level, cause-aware protection loop**:

```text
Battery Stress + Driving Context + Road Context
                  |
                  v
          Cause Attribution
                  |
                  v
     Adaptive Battery Protection
```

Acoustic road sensing is positioned as an enabling modality inside this architecture, not as a novel contribution on its own.

---

# 4. System Architecture

## 4.1 End-to-end data flow

```text
Sensors -> ESP32 -> Wi-Fi -> FastAPI backend
                                |
                 Sensor fusion / AI inference / event attribution
                                |
                        Protection decision
                                |
Motors <- Motor driver <- ESP32 <- Wi-Fi
```

## 4.2 Full architecture

```text
                           DRIVEGUARD-X
                                |
                                v
                       +------------------+
                       |  4WD ROBOT CAR   |
                       +---------+--------+
                                 |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
   Battery Sensors           MPU6050               INMP441
   V / I / Temp          Accel + Gyro           Road Audio (I2S)
          |                      |                      |
          v                      v                      v
   Battery Features        IMU Features          Audio Features
                                                  (FFT / MFCC)
          |                      |                      |
          v                      v                      v
     Battery AI             Driving AI              Road AI
          |                      |                      |
          +----------------------+----------------------+
                                 |
                                 v
                        Multimodal Fusion
                                 |
                                 v
                        Cause Attribution
                                 |
                                 v
                       Adaptive Protection
                                 |
                                 v
                  ESP32 (bounded PWM / direction)
                                 |
                                 v
                    L298N -> 4 x TT motors
```

## 4.3 Layered safety principle (critical)

The ESP32 **must retain local safety limits**. Remote AI is never the only protection.

| Layer | Component | Role |
|---|---|---|
| 0 | BMS, fuse, main switch | Hardware protection independent of all software |
| 1 | ESP32 local limits | Hard electrical/thermal limits, communication timeout, safe motor state |
| 2 | FastAPI adaptive protection | Context-aware *reduction* of power within bounds |

Rules:

- The backend may only **reduce or shape** allowed power within bounds the ESP32 independently validates. It must never be able to raise limits above Layer 1.
- Commands carry a sequence number and a time-to-live. Invalid, expired, or out-of-range commands are rejected.
- If communication is lost or commands are invalid, the ESP32 enters a **safe motor state** (motors stopped or ramped down to a defined safe level).
- Local thresholds are set conservatively above the BMS cutoffs after measurement **[OPEN]**.

## 4.4 Two operating modes

| Mode | Source | Purpose |
|---|---|---|
| **Simulation mode** | MATLAB/Simulink virtual sensors → FastAPI | Stage 1 development and demonstration |
| **Hardware mode** | ESP32 real sensors → FastAPI | Stage 2 data collection and live protection |

The AI, fusion, attribution, and protection layers are shared. The same backend serves both.

The ESP32 also has two firmware modes:

| ESP32 mode | Behavior |
|---|---|
| **Data-collection mode** | Streams/stores labeled raw or windowed data (including raw audio windows) for dataset building |
| **Run mode** | Sends compact telemetry/features and applies protection commands |

---

# 5. Sensing Modalities (Final Design Decision)

## 5.1 Final design: three modalities, one sensor group each

| Modality | Sensor(s) | Measures | Feeds |
|---|---|---|---|
| Battery | Voltage sensor module, ACS712 30A, DS18B20 | Pack voltage, battery-side current, temperature | Battery model |
| Driving behavior | MPU6050 | 3-axis acceleration, 3-axis angular velocity | IMU model |
| Road condition | INMP441 (I²S) | Acoustic signature of tire-road interaction | **Audio road model (audio-only)** |

## 5.2 Road model is audio-only

> **Design decision:** the road condition model uses **only the INMP441 microphone**. It does not take IMU features.

Rationale:

1. **Clean multimodality.** Battery, inertial, and acoustic sensing are independent observations. No sensor is shared between two models.
2. **Cleaner attribution.** Driver evidence (IMU) and road evidence (audio) come from different sensors, so the attribution engine can reason about driver-caused vs road-caused stress without both being derived from the same measurement.
3. **Clear experimental questions.** Each model can be evaluated on its own sensor.

## 5.3 Role of the IMU

The MPU6050 is the **driving behavior** sensor. Its raw data is still logged in full for every run (this keeps future analysis possible), but it is not an input to the road model.

> Note: an earlier working note listed "MPU6050 vibration and motion data" under the road modality. That wording is superseded by this decision.

## 5.4 Why the sensors are separated by role

| Question | Answered by |
|---|---|
| What is happening electrically and thermally? | Battery sensors |
| How is the vehicle being driven? | IMU |
| What is the road surface like? | Microphone |

---

# 6. Physical Prototype: 4WD Robot Car

## 6.1 Objective

The prototype is a controlled experimental platform, not a realistic passenger EV. It must let us reproduce and record:

- forward, reverse, left and right turns,
- different driving speeds and motor loads,
- different surfaces and road vibrations,
- battery voltage, current, and temperature,
- acceleration, angular velocity,
- acoustic information captured near a wheel,
- battery-stress events and protective responses,
- sensor data transmission to a backend and receipt of protection commands.

## 6.2 Chassis kit **[CONFIRMED purchased]**

Robocraze 4WD Smart Robot Car Chassis Kit (screenshot showed approximately ₹566). Kit contents:

- acrylic frame,
- four yellow TT geared DC motors,
- four wheels,
- motor brackets,
- encoder discs,
- 4×AA battery holder,
- standoffs and assembly hardware.

> **Do not use the supplied AA holder in place of the planned lithium-ion pack.**

## 6.3 Steering: differential drive

**[CONFIRMED]** Differential drive, not servo steering.

| Motors | L298N channel |
|---|---|
| Left-front + left-rear | Channel A (OUT1/OUT2) |
| Right-front + right-rear | Channel B (OUT3/OUT4) |

| Maneuver | Left side | Right side |
|---|---|---|
| Straight forward | Forward, similar speed | Forward, similar speed |
| Turn left | Slower | Faster |
| Turn right | Faster | Slower |
| Reverse | Reverse | Reverse |
| Rotate in place | Opposite direction | Opposite direction |

---

# 7. Hardware Bill of Materials and Status

The items below appeared in shopping-cart screenshots or subsequent discussions. **Do not assume every specification has been independently verified.**

| Component | Selection / observed details | Status |
|---|---|---|
| ESP32 | Robocraze ESP32 Wi-Fi/Bluetooth development board, 38-pin | Selected as replacement for an earlier ESP32 that did not work **[VERIFY exact board]** |
| Chassis | Robocraze 4WD Smart Robot Car Chassis Kit | Purchased |
| Motors | 4 × yellow TT geared DC motors (with chassis) | Purchased |
| Motor driver | L298N dual H-bridge module | Purchased |
| Battery cells | 4 × VIPOW 18650, labelled 2400 mAh, 3.7 V | Photo supplied; **discharge rating and authenticity unverified** |
| BMS | 2S balanced 20 A board (Robocraze and Electrobot HW-391 options were discussed) | **Exact purchased model and terminal layout need confirmation [OPEN]** |
| Voltage sensor | 0–25 V voltage sensor module | Purchased |
| Current sensor | ACS712 30 A module | Purchased |
| Temperature sensor | DS18B20 module | Purchased |
| Motion sensor | MPU6050 / GY-521 | Purchased |
| Microphone | INMP441 I²S MEMS microphone | Purchased |
| Display | SSD1306 0.96" 128×64 I²C OLED | Part of intended design; purchase status unconfirmed |
| Charger | TP5100 1S/2S charging module | Proposed; exact model and purchase status unconfirmed |
| Motor converter | Regulated 6 V high-current buck converter | Required; exact model/rating unconfirmed |
| Electronics converter | Regulated 5 V buck converter | Required; exact model/rating unconfirmed |
| Safety components | Fuse, holder, main switch, wires, connectors, insulation, battery enclosure | Required; final selections unconfirmed |
| Pull-up resistor | 4.7 kΩ for DS18B20 | Required only if the module lacks one |

Note: an earlier cart screenshot showed an item described as an "18650 12V 10A Lithium Protection Board." Its specifications are ambiguous. **Do not assume it is suitable for this battery.**

Policy: **prefer components already purchased.** Recommend replacements only for a genuine compatibility, performance, or safety reason. Do not buy duplicates without checking what has already been purchased.

---

# 8. Battery Pack, BMS, and Charging

## 8.1 Planned pack: 4 × 18650 in 2S2P **[PLANNED]**

```text
        Group 1 (2 cells in parallel)        Group 2 (2 cells in parallel)

        +----[Cell A]----+                    +----[Cell C]----+
 B- ----|                |---- midpoint ------|                |---- B+
        +----[Cell B]----+                    +----[Cell D]----+
```

- Two cells in parallel form group 1.
- Two cells in parallel form group 2.
- The two groups are connected in series.

| Property | Expected value (assuming suitable, matched cells) |
|---|---|
| Number of cells | 4 |
| Nominal voltage per cell | 3.7 V |
| Pack nominal voltage | 7.4 V |
| Fully charged pack voltage | 8.4 V |
| Labelled capacity per cell | 2400 mAh |
| Theoretical pack capacity | 4800 mAh (calculated from labels, **not verified**) |
| BMS type | 2S lithium-ion |
| Preferred BMS feature | Cell balancing |
| Previously discussed BMS rating | 20 A stated continuous discharge |

## 8.2 Critical concerns

- The VIPOW cell photograph does **not** establish actual capacity, authenticity, condition, or safe continuous discharge current.
- **A 20 A BMS rating does not mean the cells can safely deliver 20 A.**
- Do not assume these cells can safely power all four motors until their discharge capability is verified.
- All four cells must be matched in type, condition, capacity, and charge state.
- Do not use damaged, swollen, leaking, or overheated cells.
- **Do not energize the pack until the exact BMS terminals and battery arrangement are confirmed.**

## 8.3 Pack assembly rules **[VERIFY]**

- Before connecting two cells in parallel, **bring them to nearly the same voltage** (a small difference, for example on the order of 0.1 V or less, is commonly recommended). A large difference causes a high equalization current when they are first joined.
- Use a proper battery holder or a professionally spot-welded pack. **Do not solder directly to loose lithium-ion cells unless properly equipped and trained.**
- Insulate all exposed connections. Use a secure enclosure.
- In 2S2P the BMS "sees" each parallel pair as one cell; the BMS balance lead (B1/BM) connects to the **midpoint between the two series groups**.
- Verify cell capacity with a suitable charger/tester and verify the discharge rating from the cell datasheet or a trustworthy test. Treat the label as unverified.

## 8.4 BMS decision and verification

Two options were discussed (neither is assumed to be the board already owned):

| Option | Link |
|---|---|
| A: Robocraze 2-String 20A Lithium Battery Protection Module, Balanced Version | https://robocraze.com/products/2-string-20a-lithium-battery-protection-module-balanced-version |
| B: Electrobot HW-391, 2S 7.4 V / 8.4 V 20 A balanced BMS | https://store.electrobot.co.in/product/bp220 |

Required BMS characteristics:

- designed for 2S lithium-ion packs, 7.4 V nominal / 8.4 V full,
- cell balancing,
- overcharge, over-discharge, overcurrent, and short-circuit protection,
- adequate rating for the actual cells, motors, converters, and wiring.

Before wiring, verify:

- the exact model and specifications,
- that it is truly a 2S **balanced** BMS,
- terminal markings (B−, B1/BM, B+, P−, P+, as applicable),
- the manufacturer's wiring instructions,
- whether charging and discharging share the same port.

**Do not assume inexpensive BMS boards share terminal layouts. Do not purchase another BMS automatically; identify the existing board first.**

## 8.5 Charging: TP5100 **[PROPOSED]**

Intended arrangement:

```text
Suitable DC adapter -> TP5100 (set to 2S / 8.4 V) -> correct BMS charging port -> 2S2P battery
```

Requirements:

- Configure the TP5100 for **2S / 8.4 V** (do not leave it in 1S / 4.2 V mode).
- Use an adapter that meets the module's input requirements (a 2S charge target of 8.4 V implies the input must sit comfortably above that; check the exact module's input range).
- Select a charge current appropriate for the cells and the charger's thermal limits.
- Keep the BMS connected according to its documentation.
- **Do not charge while operating the car.**
- **Do not leave charging unattended.** Stop if the pack heats abnormally, swells, or behaves unexpectedly.

The TP5100 is a charger, not a BMS replacement. Balancing is handled by the BMS if the board supports it.

A charge current around 1 A was suggested as a conservative starting point. For a 2P group of 2400 mAh-labelled cells that corresponds to roughly 0.5 A per cell (about 0.2C on the labelled capacity). It must still be checked against the actual cell specifications and the charger's thermal limits **[VERIFY]**.

Previously discussed supplier links (verify current stock, variant, price, solder-jumper configuration, and wiring instructions before ordering):

- https://www.robogears.in/shop/5930898-tp5100-4-2v-and-8-4v-dual-one-two-battery-protection-board-3779
- https://www.ktron.in/product/tp5100-lithium-battery-charging-module/
- https://www.flyrobo.in/tp5100-2a-4.2v-and-8.4v-dual-single-lithium-battery-charger-module

**Do not finalize charging pin-to-pin wiring until the exact TP5100 and BMS are identified.**

## 8.6 Measurement limitations of the planned sensing

- Only **pack-level** voltage is measured. Individual group/cell voltages are not available to the ESP32 (the BMS monitors them internally).
- The temperature sensor measures **one location**. Its placement (against the cell surface, insulated from ambient air) strongly affects what it reports.
- Current is measured on the **battery side**, which is what matters for battery stress, but this differs from motor-side current when a buck converter sits in between (Section 9.4).

---

# 9. Power Distribution

## 9.1 Planned arrangement **[PLANNED]**

```text
Battery pack (2S2P)
      |
   Verified BMS
      |
   Fuse -> Main switch
      |
   ACS712 (series, total battery load)
      |
      +-------------------------------+
      |                               |
      v                               v
 MOTOR BRANCH                  ELECTRONICS BRANCH
 6 V high-current buck         5 V buck converter
      |                               |
      v                               v
    L298N motor supply          ESP32 5V/VIN input
      |                         + sensors + OLED
      v
  4 x TT motors
```

## 9.2 Motor branch

- Protected battery supply, fuse, appropriately rated wiring.
- ACS712 in series if measuring total battery load current.
- Regulated **6 V** high-current buck converter.
- L298N motor supply, then the four TT motors.

An 8–10 A continuous-rated converter is **only a starting candidate**. Choose it from measured motor current and startup surges, and verify its real performance and thermal limits **[OPEN]**.

## 9.3 Electronics branch

- Protected battery supply.
- Regulated **5 V** buck converter with stable output and sufficient current.
- ESP32 5V/VIN input, **according to the exact board** **[VERIFY]**.
- Sensors and OLED.

## 9.4 Battery-side vs motor-side current

The ACS712 sits on the battery side. With a buck converter between battery and motors, an approximate relationship is:

```text
I_battery  ≈  (V_motor_rail × I_motor_rail) / (efficiency × V_battery)
```

So battery-side current is lower than motor-rail current when the converter steps voltage down from about 7.4–8.4 V to 6 V (and also includes the electronics branch). This matters when interpreting current thresholds, and when comparing simulation to hardware.

## 9.5 Electrical rules

- **Never connect the 7.4–8.4 V battery directly to the ESP32 3.3 V pin.**
- Verify the correct power input pin on the exact ESP32 board.
- **Never route motor current through a solderless breadboard.**
- Use appropriately rated wires and connectors.
- Join grounds where required so the ESP32 and L298N share a signal reference.
- Keep motor wiring away from the microphone and other sensitive sensor wiring.
- Do not connect multiple 5 V sources together without checking the circuitry.
- **Measure buck converter outputs before connecting electronics or motors.**
- Check the L298N regulator jumper and board documentation before supplying its logic rail (Section 10.4).
- Size the fuse and wire gauge from measured current **[OPEN]**.

---

# 10. Motor Control

## 10.1 Why not relays

Ordinary relays are not suitable as the primary motor driver: they cannot provide rapid PWM speed control and need extra circuitry for reversal.

## 10.2 L298N **[CONFIRMED purchased]**

The L298N is already owned. **Test it before recommending a replacement.**

| Item | Value / note |
|---|---|
| Channel A | Left motors (front + rear in parallel) |
| Channel B | Right motors (front + rear in parallel) |
| Enable | ENA / ENB driven by ESP32 PWM (remove the ENA and ENB jumpers) |
| Direction | IN1/IN2 (left), IN3/IN4 (right) |
| Motor supply | Regulated 6 V |

## 10.3 Known limitations of the L298N **[VERIFY by measurement]**

- It has a **significant voltage drop** across its bipolar output stage, so motors receive noticeably less than the supply voltage.
- It dissipates that loss as heat.
- Two motors in parallel per channel may exceed its comfortable current capacity during startup or heavy load.

Actions:

- Measure the TT motors' actual current (free-running and under a moderate, non-stalled load).
- **Do not deliberately stall the motors.**
- Watch L298N and converter temperatures.
- Because the drop reduces motor voltage, tune the 6 V rail only within the motor's rated voltage range after measuring.

If the L298N cannot handle the measured load, recommend a MOSFET driver. Two **BTS7960/IBT-2** modules (one per side) were discussed as a possible alternative. **Verify their actual current ratings and minimum supply voltage before recommending purchase.**

## 10.4 L298N logic supply caution **[VERIFY]**

The L298N's onboard 5 V regulator is normally intended for a higher motor-supply voltage than a 6 V rail provides, so it may not regulate properly at 6 V. Before powering the module:

- read the exact module's documentation and jumper behavior,
- decide whether the logic rail should come from the **5 V electronics converter** (with the onboard regulator jumper handled accordingly) instead of the onboard regulator,
- keep grounds common,
- avoid back-feeding two 5 V sources into each other.

Resolve this before the first powered test.

## 10.5 Driver input source and protection shaping **[OPEN design item]**

The car needs a source of *driver requests* so that "driver behavior" can be reproduced:

| Option | Notes |
|---|---|
| Manual input (web joystick/dashboard/phone) | Natural, but less repeatable |
| **Scripted driving profiles on the ESP32/backend** | **Recommended for experiments**: repeatable normal vs aggressive acceleration/braking |

Protection should then act as a **bounded shaper on the requested PWM**:

```text
final_pwm = min(driver_requested_pwm, protection_pwm_limit)   (then rate-limited by ramp)
```

with a ramp (slew-rate) limit so protection does not itself create abrupt load changes. Specific limits are determined experimentally.

---

# 11. ESP32 Board and Pin Map

## 11.1 Board

The earlier ESP32 was not working. A **Robocraze 38-pin ESP32 Wi-Fi/Bluetooth development board** was considered as a replacement. It should be suitable if it is a conventional ESP32 dev board exposing the required GPIOs and supporting I²C, I²S, ADC, PWM, and Wi-Fi.

Before integration:

1. Confirm the exact board and module.
2. Use a known-good **USB data** cable.
3. Upload a basic Blink program.
4. Confirm serial output.
5. Test Wi-Fi.
6. Verify the actual pin labels.

Do not assume a replacement board fixes a problem actually caused by a bad cable, drivers, firmware, board selection, or wiring.

## 11.2 Proposed GPIO map **[PROPOSED; keep unless the actual board requires changes]**

| Component | Signal | ESP32 GPIO |
|---|---|---:|
| DS18B20 | DATA | GPIO4 |
| MPU6050 | SDA | GPIO21 |
| MPU6050 | SCL | GPIO22 |
| OLED SSD1306 | SDA | GPIO21 |
| OLED SSD1306 | SCL | GPIO22 |
| INMP441 | SD | GPIO23 |
| INMP441 | SCK/BCLK | GPIO18 |
| INMP441 | WS/LRCL | GPIO19 |
| ACS712 | Analog output (safely scaled) | GPIO34 |
| Voltage sensor | Analog output (safely scaled) | GPIO35 |
| L298N | ENA / PWM left | GPIO25 |
| L298N | IN1 | GPIO26 |
| L298N | IN2 | GPIO27 |
| L298N | ENB / PWM right | GPIO32 |
| L298N | IN3 | GPIO33 |
| L298N | IN4 | GPIO14 |

## 11.3 Pin notes

- GPIO34 and GPIO35 are **input-only** and suitable for ADC inputs.
- ADC1 pins are preferable for analog sensors while Wi-Fi is active on a classic ESP32 (ADC2 conflicts with Wi-Fi). GPIO34/35 are ADC1.
- **Calibrate ADC readings.** The ESP32 ADC is nonlinear and noisy; use averaging/oversampling and a calibration against a multimeter.
- The OLED and MPU6050 share the I²C bus. **Scan the bus to confirm addresses** rather than assuming them.
- **Boot-time behavior [VERIFY]:** some ESP32 pins (GPIO14 is a commonly cited one) can output a signal or glitch during boot/reset. Because these pins drive motor-driver inputs, add **pull-down resistors** on ENA, ENB, and IN1–IN4 so the motors stay off during boot, reset, and firmware upload. Verify against the exact board.
- Confirm that the I²S pins (GPIO18/19/23) are free on the exact board and not shared with onboard peripherals.

---

# 12. Component-Level Wiring

> All tables below are for planning. **State which connections require verification against the exact board before connecting power.**

## 12.1 System ground and power overview

```text
          BATTERY PACK (2S2P) + BMS
                    |
              Fuse + Main switch
                    |
                 ACS712 (IP+ / IP-)
                    |
        +-----------+-----------+
        |                       |
   6 V BUCK (motors)        5 V BUCK (electronics)
        |                       |
   L298N motor +12V pin     ESP32 5V/VIN
        |                   Sensors / OLED (3.3 V from ESP32 regulator)
   4 x TT motors

   All grounds must meet at a common reference (star-point preferred).
```

## 12.2 L298N

| L298N terminal | Connection |
|---|---|
| OUT1 / OUT2 | Both left motors in parallel |
| OUT3 / OUT4 | Both right motors in parallel |
| ENA | GPIO25 |
| IN1 | GPIO26 |
| IN2 | GPIO27 |
| ENB | GPIO32 |
| IN3 | GPIO33 |
| IN4 | GPIO14 |
| Motor supply | Regulated 6 V output |
| GND | Common ground |

- **Remove the ENA and ENB jumpers** when using ESP32 PWM.
- Check motor lead polarity and mounting orientation. **Test with the wheels lifted** and correct wiring or firmware if a side runs backward.
- Add pull-downs on enable/input lines (Section 11.3).

## 12.3 DS18B20

| Pin | Connection |
|---|---|
| VCC | 3.3 V |
| GND | Common ground |
| DATA | GPIO4 |

Use a **4.7 kΩ pull-up** from DATA to 3.3 V if the module does not already include one.

## 12.4 MPU6050 / GY-521

| Pin | Connection |
|---|---|
| VCC | 3.3 V, unless the exact module documentation says otherwise |
| GND | Common ground |
| SDA | GPIO21 |
| SCL | GPIO22 |

Mount **rigidly** to the chassis and **document orientation**.

## 12.5 INMP441

| Pin | Connection |
|---|---|
| VDD | 3.3 V |
| GND | Common ground |
| SCK/BCLK | GPIO18 |
| WS/LRCL | GPIO19 |
| SD | GPIO23 |
| L/R | GND (selects one channel), if required |

Mount near the selected wheel or wheel housing, protected from impact, with a **fixed position across all experiments**. Keep the wiring away from motor wiring.

## 12.6 OLED SSD1306

| Pin | Connection |
|---|---|
| VCC | 3.3 V if supported |
| GND | Common ground |
| SDA | GPIO21 |
| SCL | GPIO22 |

## 12.7 Voltage sensor module

Connect the sensing terminals across the intended battery measurement points. Connect the output to GPIO35 **only after confirming that the module's divider keeps the ESP32 input within its permitted range**. Follow the module's own supply and terminal markings.

## 12.8 ACS712

The ACS712 has high-current terminals **IP+ / IP−** and electronics terminals **VCC / GND / OUT**.

- Place IP+ and IP− **in series with the load-current path.** **Never connect the high-current terminals to the ESP32.**
- Many ACS712 modules run at 5 V and their output can exceed 3.3 V. **Verify the exact output range and use signal conditioning or a divider before GPIO34.**
- Calibrate zero-current offset and sensitivity.
- Place it before the power branches to measure **total battery load current**. Account for the charging path separately (the car must not be charged while operating).

---

# 13. Sensor Details and Calibration

## 13.1 Voltage sensor

- Typical 0–25 V modules use a resistor divider (commonly about 5:1); at a fully charged 8.4 V pack the output is then well within the ESP32's input range. **Verify the actual divider ratio with a multimeter [OPEN].**
- Calibration procedure:
  1. Compare ADC reading with a multimeter at several known voltages.
  2. Fit a linear (or piecewise) correction.
  3. Average multiple samples to reduce noise.
- Record calibration constants in the repository.

## 13.2 ACS712 (30 A variant)

- The 30 A variant typically has a sensitivity on the order of 66 mV/A and a zero-current output near half the supply voltage **[VERIFY on the actual module]**.
- A 30 A sensor has **coarse resolution at the low currents** a small robot car may draw, and its zero offset drifts and is noisy. Expect to need averaging/filtering.
- Calibration procedure:
  1. Measure zero-current output (motors off, no load through the sensor).
  2. Apply a known load current measured by a trusted meter in series.
  3. Compute offset and sensitivity.
  4. Re-check the offset at power-up (auto-zero) and after warm-up.
- If the ESP32 ADC plus ACS712 noise prove inadequate for the project's measurements, consider a better ADC or current-sense option. That would be a **genuine performance issue**, justified by measurements, not a default purchase.

## 13.3 DS18B20

- Digital 1-Wire sensor; a 12-bit conversion takes on the order of hundreds of milliseconds, so temperature updates are slow. Sample at low rate.
- **Placement:** in thermal contact with the pack surface and insulated from airflow; document the location. Battery temperature changes slowly, so a low rate is appropriate.
- Optional: additional temperature points (for example near the L298N or buck converter) are useful for safety, not required.

## 13.4 MPU6050

- Accelerometer + gyroscope over I²C; configure full-scale ranges and a digital low-pass filter appropriate for driving dynamics.
- Calibrate gyro bias at rest at startup.
- Mount rigidly; document axes relative to the vehicle's forward direction.
- Raw data is logged in full for every run, but only the driving model consumes it.

## 13.5 INMP441

- I²S MEMS microphone delivering 24-bit samples in a 32-bit frame; the ESP32 receives them over I²S (**I²S, not I²C**).
- Sampling rate: a value such as 16 kHz is a typical starting point **[PROPOSED]**; it should be chosen consistently with the public road-audio datasets used for training so features are comparable.
- Mounting: near a wheel/wheel housing, as far from the motors/gearboxes as is practical; foam windscreen; protected from impact; position documented and fixed.
- Characterize motor noise by recording **motor-only / wheels-lifted** clips.

## 13.6 OLED

- 128×64 SSD1306 on the shared I²C bus. Display pack voltage, current, temperature, driving class, road class, attributed cause, protection state, link status.

---

# 14. ESP32 Firmware Design

## 14.1 Responsibilities

1. Initialize sensors.
2. Sample battery voltage, current, and temperature.
3. Read MPU6050 acceleration and angular velocity.
4. Capture INMP441 audio (raw windows or computed features).
5. Timestamp and package observations.
6. Transmit telemetry over Wi-Fi.
7. Receive protection commands.
8. Apply **bounded** motor direction and PWM.
9. Update the OLED.
10. Detect sensor errors and communication timeouts.
11. **Enter a safe motor state** when commands are invalid or communication fails.

The ESP32 does not need to run heavy ML locally. Inference runs on the FastAPI backend, while **local safety limits stay on the ESP32**.

## 14.2 Suggested task structure **[PROPOSED]**

| Task | Rate (starting point) | Notes |
|---|---|---|
| Battery V/I sampling | Oversample and average internally; report ~10 Hz | Averaging reduces ADC/ACS712 noise |
| Temperature | ~1 Hz | DS18B20 conversion time |
| IMU | ~100 Hz internal; windowed/reported | Raw logged in data-collection mode |
| Audio (I²S + DMA) | e.g. 16 kHz continuous capture | Dedicated task; do not block on Wi-Fi |
| Telemetry TX | ~ few Hz to 10 Hz | Compact in run mode |
| Command RX / safety check | Every loop | Timeout watchdog |
| OLED | ~2–5 Hz | Low priority |
| Motor control + ramp | High priority, fixed period | Applies limits and ramping |

Use FreeRTOS tasks (the ESP32 has two cores) so audio capture and motor safety are not starved by Wi-Fi activity.

## 14.3 Audio handling

- **Data-collection mode:** stream or log raw audio windows with timestamps and run IDs to the backend/laptop for labeled dataset building.
- **Run mode:** compute compact audio features (for example RMS and band energies) on the ESP32 and/or send short windows, with heavier extraction (MFCC etc.) on the backend. Raw audio at ~16 kHz × 16-bit is roughly tens of kB/s, which Wi-Fi can carry, but it competes with other traffic; design the split deliberately.

## 14.4 Local safety state machine **[PROPOSED]**

```text
              +-----------+
   power-up ->|   BOOT    |  motors forced off, outputs low
              +-----+-----+
                    |  self-checks pass
                    v
              +-----------+   comms timeout / bad command
              |  NORMAL   |------------------------------+
              +-----+-----+                               |
                    | local limit exceeded                v
                    v                              +-------------+
              +-----------+                        | SAFE / STOP |
              | LIMITING  |----------------------->|  (ramp down)|
              +-----------+     fault persists     +------+------+
                                                          |
                                         faults cleared + operator reset
                                                          v
                                                      NORMAL
```

Local triggers (thresholds to be set from measurement, conservatively above BMS cutoffs):

- battery-side overcurrent,
- pack undervoltage,
- over-temperature,
- sensor failure (invalid/stuck readings),
- communication timeout,
- invalid or expired command.

## 14.5 Missing-sensor behavior (documented)

| Missing/failed input | System behavior |
|---|---|
| Audio (INMP441) | Road class = `unknown`; attribution confidence reduced; operate on battery + IMU evidence |
| IMU (MPU6050) | Driving class = `unknown`; attribution confidence reduced |
| Temperature | Treat as a safety fault → conservative local limit |
| Voltage or current | Safety fault → conservative local limit or safe state |
| Backend/Wi-Fi | ESP32 local limits only → safe motor state after timeout |

---

# 15. Communication Protocol

Transport: Wi-Fi, HTTP/REST (JSON) to start. A different transport (WebSocket, MQTT) can be evaluated later if latency or overhead requires it.

## 15.1 Telemetry (ESP32 → backend) **[PROPOSED]**

```json
{
    "device_id": "dgx-car-01",
    "seq": 10234,
    "t_ms": 123456,
    "mode": "run",
    "battery": { "v_pack": 7.62, "i_batt": 2.81, "temp_c": 31.4 },
    "imu": { "ax": 0.2, "ay": 0.1, "az": 9.81, "gx": 0.03, "gy": 0.02, "gz": 0.04 },
    "audio": { "window_ms": 1000, "rms": 0.12, "spectral_centroid": 2300, "mfcc": [] },
    "motor": { "pwm_left": 180, "pwm_right": 180, "dir_left": 1, "dir_right": 1 },
    "faults": []
}
```

## 15.2 Command (backend → ESP32) **[PROPOSED]**

```json
{
    "seq": 5521,
    "ack_seq": 10234,
    "ttl_ms": 500,
    "action": "limit_power",
    "pwm_limit_pct": 75,
    "reason": "driver_induced",
    "confidence": 0.82
}
```

Rules:

- `pwm_limit_pct` is bounded; the ESP32 re-validates it and clamps it.
- Commands expire after `ttl_ms`. After expiry or timeout the ESP32 falls back to local limits / safe state.
- Out-of-order or duplicate `seq` values are ignored.
- The ESP32 may apply **stricter** local limits than the command, never looser.
- Values shown are illustrative, **not experimental results**.

## 15.3 Validation on the backend

- Check timestamp monotonicity and plausible ranges for every measurement.
- Reject or flag NaN/out-of-range values and note the sensor fault.
- Store raw telemetry, model outputs, attribution, and commands for later analysis.

---

# 16. AI Models

Three models, each fed by exactly one sensor group:

```text
Battery sensors  -> Battery features -> Battery model
MPU6050          -> IMU features     -> Driving model
INMP441          -> Audio features   -> Road model (audio-only)
```

## 16.1 Model 1: Battery model

### Objective

Estimate battery condition and/or stress. Depending on what the selected dataset supports and what can be validated reliably, outputs may include:

- State of Health (SOH),
- Remaining Useful Life (RUL),
- battery stress score,
- battery risk score,
- degradation state.

> Do not force RUL if the dataset does not support it well. A reliable SOH/stress model is more valuable than an unreliable RUL model.

### Initial dataset strategy

Train first on public lithium-ion datasets (the RC pack will not have enough aging data at the start):

- NASA Battery Dataset
- Oxford Battery Degradation Dataset
- other public battery-aging datasets

Pipeline:

```text
Dataset -> EDA -> Preprocessing -> Feature Engineering -> Training -> Evaluation -> Saved Model
```

### Candidate algorithms

- Random Forest
- XGBoost

XGBoost is a strong candidate (nonlinear relationships, engineered features, lightweight, easy Python deployment, inspectable feature importance), but the final choice is made from experiments.

### Features (dataset dependent)

Voltage, current, temperature, SOC, cycle number, charge/discharge duration, mean voltage and variation, mean current and variation, temperature rise and slope, voltage slope, rolling mean and standard deviation, charge/discharge capacity, internal-resistance-related features where available. Time-window features: mean, std, min/max, slope, rate of change.

### Domain shift (important)

A model trained on NASA/Oxford data is not automatically a model for this **2S2P, VIPOW-labelled 18650 pack**. Differences include cell chemistry, capacity, internal resistance, age, charge/discharge rate, temperature, cell count, and sensor characteristics (ESP32 ADC + voltage sensor + ACS712).

```text
Public Battery Dataset -> Initial Model -> RC Prototype -> Collect RC Battery Data
        -> Evaluate Domain Shift -> Retrain / Calibrate -> Target-Domain Battery Model
```

For XGBoost / Random Forest this is **retraining, recalibration, or target-domain adaptation**, not neural-network fine-tuning.

## 16.2 Model 2: IMU / driving behavior model

### Objective

Classify **how the vehicle is being driven** from the MPU6050.

Candidate classes (final set depends on dataset and reproducible RC maneuvers):

- Normal
- Aggressive acceleration
- Aggressive braking
- Sudden maneuver

### Features

Mean, standard deviation, variance, RMS, max/min, peak-to-peak, acceleration magnitude, gyroscope magnitude, **jerk** (da/dt), angular-velocity statistics, longitudinal/lateral acceleration characteristics, low-frequency energy, rolling statistics.

### Separating driving dynamics from road vibration

A rough road will still shake the IMU, even though the IMU is not the road sensor. If ignored, the model may mistake road vibration for aggressive driving and corrupt attribution. Mitigation:

- emphasize **low-frequency, longitudinal/lateral** features reflecting driver inputs,
- **low-pass / band-limit** the IMU signal before feature extraction,
- include **rough-road runs with a normal driver** in evaluation,
- report this as an explicit robustness check.

### Pipeline

```text
Public IMU Dataset -> Cleaning -> Windowing -> Feature Extraction -> Training -> Validation/Testing -> Saved Model
```

Candidates: Random Forest, XGBoost, SVM, and a lightweight neural network only if the data supports it. Classical ML first.

## 16.3 Model 3: Audio road model (audio-only)

### Objective

Estimate road-surface condition from the **INMP441 only**.

```text
INMP441 -> Preprocessing (filter, normalize, window) -> FFT + MFCC -> Audio features -> Road model -> Road condition
```

### Why a microphone

Different surfaces produce different tire-road interaction sounds: a smooth surface gives a relatively stable acoustic pattern; rough surfaces change the spectrum and energy; gravel adds distinct high-frequency/irregular activity; a bump produces a short impact event. Tire-road acoustic sensing and road-surface classification already exist in the literature, so the project does **not** claim acoustic road sensing as novel.

### Audio features

RMS energy, spectral centroid, bandwidth, roll-off, zero-crossing rate, dominant frequency, frequency-band energy, **MFCC**, spectral entropy, short-time energy variation (useful for impact events). If a neural network is later justified, log-mel spectrograms are an alternative input; classical features come first.

### Road classes

Initial classes (final set decided from datasets and the surfaces the RC car can reproduce):

```text
1. Smooth   2. Moderately Rough   3. Rough   4. Impact Event
```

Alternative: Low / Medium / High disturbance. Do not create more classes than can be reliably collected and labeled.

### Audio-only design considerations

Using one sensor for the road model keeps the system simple, but the model must be robust to anything that changes the sound without changing the road.

| Confounding factor | Why it matters | Mitigation |
|---|---|---|
| Motor / gearbox noise | TT motors are loud and can mask tire-road sound | Mount mic away from motors, record motor-only baselines, band-pass filter, include motor noise in all training runs |
| Vehicle speed | Tire noise changes with speed, not only surface | Record each surface at several controlled speeds; log PWM/speed per run; report per-speed performance |
| Wind / ambient noise | Adds broadband noise | Foam windscreen, noise augmentation, record in varied environments |
| Mic placement | Spectrum depends strongly on position | Fix position for the whole project; document it |
| Impact events are short | Long windows smear a brief bump | Shorter windows or a dedicated short-window impact detector |
| Public dataset domain | Public data comes from full-size vehicles | RC target-domain adaptation (Section 22) |
| Supply noise on the mic | Switching converters/motors can couple noise | Keep mic wiring short and away from motor/buck wiring; clean 3.3 V |

### Road model experiments

- **R1 Classifier comparison:** Random Forest vs XGBoost vs SVM (optional small CNN on log-mel).
- **R2 Feature-set ablation:** spectral-only vs MFCC-only vs combined.
- **R3 Robustness:** different speeds, added noise, motor noise present vs reduced, different recording sessions/days.
- **R4 Target-domain adaptation:** public-audio-trained vs RC-calibrated.

Report accuracy, precision, recall, F1, and confusion matrix.

## 16.4 Models and datasets summary

| Model | Input sensor | Public data to start | RC adaptation |
|---|---|---|---|
| Battery | V, I, T | NASA / Oxford / other | Recalibrate/retrain on RC pack data |
| Driving | MPU6050 | Public IMU/driving dataset | Retrain on RC IMU runs |
| Road | INMP441 | Public tire-road/road-surface audio | Retrain on RC audio runs |

Always check each public dataset's sampling rate, sensor placement, label definitions, and vehicle type before use.

---

# 17. Fusion, Attribution, and Adaptive Protection

## 17.1 Multimodal fusion

Combine the three model outputs (with confidences) into one context state:

```text
Battery:  High stress / SOH
Driving:  Aggressive | Normal | ...
Road:     Smooth | Rough | Impact | ...
```

## 17.2 Cause attribution engine (rule-based first)

The first version is intentionally **rule-based**. There is no need for another black-box model before the basic system is validated.

Inputs: battery stress, battery SOH, driving behavior (IMU), road condition (audio), confidence scores, temporal coincidence.

| Condition | Likely cause |
|---|---|
| Battery stress high + driving aggressive + road disturbance low | **Driver** |
| Battery stress high + driving normal + road disturbance high | **Road** |
| Battery stress high + driving normal + road smooth + SOH low | **Battery** |
| Several factors significant | **Mixed** |

Output: primary cause, confidence, and the evidence contributed by each modality.

Because road evidence comes from a single sensor, the engine weights it by the **road model's confidence**: a low-confidence road prediction reduces the weight of road evidence instead of being treated as fact. The same applies to the other modalities.

## 17.3 Temporal attribution (after the basic rules work)

Timing strengthens attribution:

```text
t=10.0 s  Driver accelerates (IMU)
t=10.2 s  Current rises
t=10.3 s  Battery stress increases   -> supports driver attribution

t=20.0 s  Rough-road / impact signature (audio)
t=20.1 s  Current begins to fluctuate
t=20.3 s  Battery stress event       -> supports road attribution
```

This requires **tight time synchronization** between sensor streams (Section 21.4).

## 17.4 Adaptive protection

| Attributed cause | Intent | Example response (illustrative) |
|---|---|---|
| Driver | Reduce the driver-induced load | Moderate power limitation |
| Road | Avoid unnecessary severe intervention | Continue monitoring, mild or no limit |
| Battery | Protect the degraded battery | Strong protection |
| Mixed | Multiple contributors | Strong protection |

Exact PWM limits are determined experimentally. **Do not present values such as 50%, 70%, or 75% as predetermined scientific results.** Protection is applied as a bounded, ramped cap on requested PWM (Section 10.5), always inside the ESP32's local limits.

## 17.5 Conventional baseline (required)

```text
Battery Current -> Threshold -> Normal: no intervention / High: reduce PWM
```

The research comparison is DriveGuard-X versus this baseline.

---

# 18. FastAPI Backend

## 18.1 Responsibilities

- Receive ESP32 telemetry.
- Validate timestamps and measurements.
- Store sensor data and events.
- Run or call the fusion and inference models.
- Attribute likely causes of battery-stress events where supported.
- Generate explicit protection commands.
- Return commands in a defined format.

## 18.2 Structure

```text
backend/
|-- app/
|   |-- main.py
|   |-- api/            (routes.py, schemas.py)
|   |-- models/         (battery_model.py, imu_model.py, road_model.py [audio-only])
|   |-- preprocessing/  (battery_features.py, imu_features.py, audio_features.py)
|   |-- fusion/         (fusion_engine.py)
|   |-- attribution/    (attribution_engine.py)
|   `-- protection/     (protection_engine.py)
|-- models/             (battery_model.pkl, imu_model.pkl, road_model.pkl)
|-- data/
`-- requirements.txt
```

## 18.3 Inference design

Keep model-specific processing separate. Each model receives data from exactly one sensor group:

```text
Incoming data
   |-- Battery processing -> Battery model  (V/I/T)
   |-- IMU processing     -> Driving model  (MPU6050)
   `-- Audio processing   -> Road model     (INMP441, audio-only)
              |
              v
   Fusion -> Attribution -> Protection
```

## 18.4 Endpoint and response

```http
POST /api/inference
```

Illustrative response (not experimental results):

```json
{
    "battery_state": { "soh": 87.2, "risk_score": 41 },
    "driving_behavior": { "class": "aggressive", "confidence": 0.91 },
    "road_condition": { "class": "rough", "confidence": 0.84 },
    "attribution": { "cause": "driver", "confidence": 0.82 },
    "protection": { "action": "limit_power", "pwm_limit": 75 }
}
```

The `imu` block of the request feeds only the driving model; the `audio` block feeds only the road model.

## 18.5 Additional backend endpoints (suggested)

| Endpoint | Purpose |
|---|---|
| `POST /api/telemetry` | Ingest ESP32 data (run mode) |
| `POST /api/collect` | Ingest labeled data (data-collection mode) |
| `GET /api/status` | Link health, latest states |
| `POST /api/scenario` | Select simulation/experiment scenario |
| `GET /api/events` | Stress events with attribution and evidence |

---

# 19. MATLAB / Simulink Simulation

## 19.1 Purpose

Create a controlled virtual EV/RC-car environment that sends sensor-like data to the **same FastAPI backend** later used with real ESP32 data. Target complexity: **a simplified longitudinal EV/RC-car model with realistic virtual sensor signals and closed-loop protection**, not a commercial-grade EV simulator.

## 19.2 Architecture

```text
Driver model     Road model
      \             /
       v           v
      Vehicle dynamics
            |
            v
       Motor model
            |
            v
       Battery model
            |
            v
   Virtual sensors (battery, IMU, audio)
            |
            v
        FastAPI backend
   (Battery | IMU | Audio road models)
            |
            v
   Fusion -> Attribution -> Protection
            |
            v
   Protection output -> back to Simulink (closed loop)
```

## 19.3 Components

**Driver model:** normal and aggressive driver; inputs throttle and brake; produces torque/power demand.

**Motor model:** simplified DC motor: throttle → torque → wheel force → acceleration, with current demand. Goal is realistic cause-and-effect, not exact commercial modeling.

**Vehicle dynamics:** simplified longitudinal equation:

```text
m * dv/dt = F_traction - F_drag - F_rolling - F_road
```

**Road model:** scenarios Smooth, Rough, Bump, Pothole-like disturbance, Uneven surface. The scenario affects vehicle dynamics, battery load, and the virtual audio, so mechanical disturbance and acoustic evidence stay consistent.

**Battery model:** simplified:

```text
V_battery = V_oc(SOC) - I * R_internal
Heat ∝ I² R
```
Outputs voltage, current, SOC, temperature, SOH, stress/risk. Can be refined later. Parameters should eventually reflect the measured RC pack.

**Virtual IMU:** generates ax, ay, az, gx, gy, gz with sensor noise. Because the IMU model is for driving behavior only, the virtual IMU should still contain realistic road-induced vibration so the driving model can be tested for **ignoring road vibration**.

**Virtual road audio:**

- **Option A (preferred): real audio replay.** Select a real recording matching the scenario, extract features, send them to FastAPI. The replayed audio must be **time-aligned with simulated road events** (a simulated bump should coincide with an impact in the audio), otherwise temporal-attribution tests are meaningless.
- **Option B: synthetic acoustic features**, only as a temporary mechanism when synchronized audio is unavailable. The physical system will later use live INMP441 audio.

## 19.4 MATLAB ↔ FastAPI

HTTP/REST: MATLAB/Simulink sends an HTTP POST to FastAPI and receives a JSON response, which is applied as the protection output to close the loop.

## 19.5 Simulation vs hardware differences to keep in mind

- Simulation currents are idealized; hardware has L298N drop, buck-converter conversion (battery-side vs motor-side current), ADC noise, and ACS712 offset.
- The simulation should expose the **battery-side** current to match the ACS712 measurement point.
- Parameters should be recalibrated after RC measurements are available.

---

# 20. Dashboard

Designed like a digital EV instrument cluster (Next.js/React, real-time).

```text
+--------------------------------------------+
|               DRIVEGUARD-X                 |
|                                            |
| SPEED       SOC        SOH       TEMP      |
| 32 km/h     78%        87%       34 C      |
|                                            |
| CURRENT     VOLTAGE    POWER               |
| 3.2 A       7.5 V      24 W                |
|                                            |
| DRIVER:     AGGRESSIVE                     |
| ROAD:       SMOOTH                         |
|                                            |
| BATTERY RISK: HIGH                         |
|                                            |
| CAUSE: DRIVER-INDUCED                      |
| CONFIDENCE: 82%                            |
|                                            |
| PROTECTION: ACTIVE                         |
| PWM LIMIT: 75%                             |
+--------------------------------------------+
```

(Values are illustrative.)

Live graphs: current, voltage, temperature, acceleration, battery risk, road condition / audio disturbance level, protection action, link status and fault flags.

**Simulation control panel:** scenario buttons (Normal, Aggressive Driver, Rough Road, Degraded Battery, Aggressive + Rough, Combined Stress) whose selection is sent to MATLAB/Simulink, for clean demonstrations during reviews. A corresponding panel can trigger scripted driving profiles in hardware mode.

---

# 21. Data Collection, Labeling, and Leakage Control

## 21.1 Principles

- Collect **controlled, labeled** runs. Do not drive randomly and record an unlabeled stream.
- Each recording corresponds to a known scenario.
- Keep raw data so features can be redesigned without repeating physical experiments (especially valuable for audio).

Example labels:

```text
Run_001  Road=Smooth  Driver=Normal
Run_002  Road=Rough   Driver=Normal
Run_003  Road=Gravel  Driver=Normal
Run_004  Road=Bump    Driver=Normal
Run_101  Driving=Normal
Run_102  Driving=Aggressive Acceleration
Run_103  Driving=Hard Braking
```

## 21.2 Combined scenarios

After individual scenarios work:

```text
Normal Driver + Smooth / Rough / Gravel / Bump
Aggressive Driver + Smooth / Rough
Normal Driver + Degraded Battery
Aggressive Driver + Degraded Battery
Aggressive Driver + Rough Road + Degraded Battery
```

These validate the attribution engine.

## 21.3 Logical record

```text
timestamp, battery_voltage, battery_current, temperature,
accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
audio_reference, pwm_command, scenario, driver_label, road_label
```

Logging `pwm_command` lets us analyze how speed/power affects the audio model.

## 21.4 Synchronization

All streams share a common timestamp assigned consistently by the ESP32. Synchronization matters because attribution depends on whether a current spike, a driver acceleration event (IMU), and a road disturbance (audio) occurred at approximately the same time. Record synchronization accuracy and any known latency between I²S audio capture and other sensors.

## 21.5 Time windows

Models operate on windows, not isolated samples (for example 2–5 s, chosen experimentally). Different models likely use different windows; the audio road model may use shorter windows or a short-window impact detector.

## 21.6 Avoiding time-series leakage

Overlapping windows leak information across train/test splits. **Split at the run level:**

```text
Run 1 -> Train    Run 2 -> Train    Run 3 -> Validation    Run 4 -> Test
```

For audio, additionally keep whole **recording sessions** (same day/location/conditions) in one split, since they share background-noise characteristics.

## 21.7 Road experiment protocol

For each surface: start recording → drive at approximately controlled speed → complete route → stop recording → save run ID → record road label → record speed/driver condition → repeat several times (for example 10 runs per surface, as time allows).

Because the road model depends only on audio:

- record each surface at **several controlled speeds**,
- repeat some runs in **different sessions/conditions**,
- record **motor-only / wheels-lifted baselines**.

## 21.8 Driving experiment protocol

Controlled throttle/brake patterns: normal acceleration, aggressive acceleration, normal braking, hard braking, constant speed, repeated acceleration. Repeat on both smooth and rough surfaces so the IMU model can be checked for road-vibration sensitivity.

## 21.9 Battery experiment protocol

Collect across controlled states: high SOC/healthy, medium SOC/healthy, lower SOC/healthy, aged or degraded condition.

> **Safety:** deliberately degrading lithium-ion cells is hazardous. For the academic prototype, battery-health labels should initially rely on controlled electrical characterization and/or the public dataset rather than rapidly manufacturing severe degradation. **Do not abuse cells.**

## 21.10 Data layout

```text
data/
|-- raw/        (battery/, imu/, audio/)
|-- processed/  (battery_features.csv, imu_features.csv, audio_road_features.csv)
`-- labels/     (road_labels.csv, driving_labels.csv, scenario_labels.csv)
```

---

# 22. Domain Shift, Adaptation, and Augmentation

## 22.1 Why domain shift matters

The same surface or event looks different on different vehicles: mass, wheel size, tire material, suspension, ground clearance, motor noise/vibration, sensor mounting, microphone location, speed, chassis. A small plastic-wheeled RC car sounds very different from a full-size vehicle on the same surface. Public data gives a general starting point; RC data adapts it to this platform.

## 22.2 Strategy

```text
Stage 1: Public dataset -> preprocessing -> features -> initial model
Stage 2: RC car + ESP32 + sensors -> labeled RC data -> retrain/calibrate -> RC model
```

## 22.3 Adaptation procedure

1. Evaluate the public-data model on RC data (baseline target-domain performance).
2. Identify errors (for example smooth classified as rough; motor noise mistaken for surface noise).
3. Collect additional examples for difficult cases.
4. Normalize/standardize features consistently across public and RC data.
5. Retrain or recalibrate on combined data.
6. Evaluate on a **held-out RC test set**.
7. Compare before vs after adaptation (this becomes a paper result).

## 22.4 Transfer learning

- Classical ML: public data + RC data → retraining/recalibration.
- Neural networks (only if actually used, e.g. a CNN on log-mel for audio): pretrain on public data, freeze early layers, train on RC data.
- Do not add neural networks just to claim transfer learning.

## 22.5 Augmentation

| Model | Augmentations |
|---|---|
| IMU | Sensor noise, small bias changes, magnitude scaling, time shifts, small temporal distortions |
| Audio | Background noise, motor-noise mixing, amplitude/gain changes, time shifts, controlled environmental noise |

Augmentation is especially important for the audio model because it is the only road sensor. **It supplements and never replaces real RC data.**

---

# 23. Experiments and Evaluation

## 23.1 Model-level metrics

- Classifiers: accuracy, precision, recall, F1, confusion matrix.
- Regression (SOH): MAE, RMSE, R².

## 23.2 System-level metrics (most important)

| Metric | Meaning |
|---|---|
| False protection interventions | Power reduced when strong intervention was not necessary |
| Missed protection events | System failed to protect when conditions required it |
| Intervention latency | Time from meaningful stress event to reaction |
| Protection duration | How long power is limited |
| Battery stress | Stress indicators vs threshold baseline |
| Vehicle performance | Acceleration, average speed, power limitation, energy consumption, time in restricted operation |

## 23.3 Research experiments

**Experiment 1: Battery model.** Random Forest vs XGBoost; MAE, RMSE, R², error distribution.

**Experiment 2: Driving model.** Normal vs aggressive acceleration vs aggressive braking vs others; accuracy, precision, recall, F1, confusion matrix; **robustness to road vibration** (normal driver on rough road).

**Experiment 3: Road model (audio-only).** Classifier comparison; feature-set ablation; robustness (speed, noise, session).

**Experiment 4: Target-domain adaptation.** Public-trained vs RC-calibrated models (battery, IMU, audio as applicable).

**Experiment 5: Protection strategy.** Threshold-based vs DriveGuard-X on false interventions, missed events, latency, battery stress, power limitation, energy consumption, driving performance. **This is the central system-level experiment.**

## 23.4 Attribution scenario experiments

| Exp | Setup | Expected (hypothesis, not result) |
|---|---|---|
| A | Healthy battery + normal driver + smooth road | Low risk, no unnecessary intervention |
| B | Healthy + aggressive driver + smooth | High transient current, driver attribution, moderate protection |
| C | Healthy + normal driver + rough road | Road attribution, avoid severe intervention |
| D | Degraded battery + normal driver + smooth | Battery attribution, strong protection |
| E | Degraded + aggressive + rough | Mixed attribution, strong protection |

## 23.5 Worked end-to-end examples

```text
1) SOH 88%, Driver aggressive, Road smooth, Current high
   Battery: high transient stress | IMU: aggressive | Audio: smooth
   -> Fusion -> Cause: Driver -> Protection: moderate power limitation

2) SOH 88%, Driver normal, Road rough, Current fluctuating
   Battery: stress event | IMU: normal | Audio: rough
   -> Fusion -> Cause: Road -> Protection: avoid unnecessary severe intervention

3) SOH low, Driver normal, Road smooth, Current high
   Battery: high stress + low SOH | IMU: normal | Audio: smooth
   -> Fusion -> Cause: Battery -> Protection: strong intervention
```

## 23.6 Evaluation honesty rules

- Include baselines for every comparison.
- Document missing-sensor behavior and test it.
- Distinguish predictions from measured facts.
- Do not claim correlation proves causation.
- Do not claim physical test results that have not been run.

---

# 24. Safety and Validation Procedure

## 24.1 Pre-power verification order

Before powering the complete system:

1. Verify the **exact BMS model and terminal markings**.
2. Confirm the cells can supply the intended current.
3. Verify battery polarity and group voltage (and that parallel cells were voltage-matched before joining).
4. Confirm TP5100 2S configuration and input requirements.
5. Check converter input ranges, outputs, and current ratings.
6. Test the ESP32 independently (Blink, serial, Wi-Fi).
7. Test sensors individually.
8. Check every GPIO signal for voltage compatibility (3.3 V max).
9. Verify ACS712 and voltage-sensor output levels.
10. Check common grounds and return paths.
11. Test the L298N **with wheels lifted**.
12. Test **one motor side at a time**.
13. Monitor motor-driver and converter temperatures.
14. **Stop** on unexpected heating, smell, noise, unstable power, or battery damage.

## 24.2 Suggested bring-up sequence

```text
 1. ESP32 alone on USB: Blink, serial, Wi-Fi
 2. I2C scan: MPU6050 + OLED addresses
 3. DS18B20 reading
 4. INMP441 I2S capture sanity check (record, inspect waveform/spectrum)
 5. Voltage sensor + ADC calibration against multimeter (bench supply first)
 6. ACS712 zero/sensitivity calibration with a known load
 7. Buck converters on a bench: measure outputs with no load, then light load
 8. L298N + one side of motors on bench supply, wheels lifted
 9. Both sides, wheels lifted, with PWM ramp and safe-state tests
10. Battery pack + BMS verification, then supervised power-up
11. Full system, wheels lifted
12. Low-speed ground tests, then experiments
```

## 24.3 Safety rules

- **Do not deliberately stall the motors** to test current.
- Do not solder directly to loose lithium-ion cells unless properly equipped and trained.
- Use a proper holder or professionally spot-welded pack, insulated connections, a correctly installed BMS, a fuse, a main switch, and a secure enclosure.
- **Do not leave charging unattended.** Stop if the pack heats abnormally, swells, or behaves unexpectedly.
- **Do not charge while operating the car.**
- Do not use damaged, swollen, leaking, or overheated cells.
- Keep a suitable fire-safe area and supervise all early tests.
- Do not declare the system safe without verification.

---

# 25. Open Items Register

Verify these before declaring the design finalized. When information is missing, ask for a photograph, product link, or printed model number.

| # | Open item | Why it matters |
|---|---|---|
| 1 | Exact replacement ESP32 model and working condition | Pin capabilities, power input pin, previous failure cause |
| 2 | Exact BMS in possession | Terminal layout, wiring, ratings |
| 3 | Whether the BMS is truly 2S, balanced, and suitably rated | Pack safety |
| 4 | BMS terminal layout and charging-port configuration | Correct charge/discharge wiring |
| 5 | Exact TP5100 model, jumper configuration, charge-current setting | 2S charging safety |
| 6 | Authenticity, condition, discharge rating of VIPOW 2400 mAh cells | Safe current capability |
| 7 | Actual TT motor current requirements | Driver, converter, fuse sizing |
| 8 | Whether the L298N can safely handle two motors per channel | Overheating/failure risk |
| 9 | L298N logic supply arrangement at a 6 V motor rail | Regulator dropout, back-feed risk |
| 10 | Motor buck converter rating and thermal capability | Startup surges, heat |
| 11 | Electronics buck converter rating | Stable 5 V for ESP32/sensors |
| 12 | Fuse rating and wire gauge from measured current | Fault protection |
| 13 | Voltage-sensor divider ratio | ADC range/calibration |
| 14 | ACS712 supply voltage and analog output range | ADC protection/scaling |
| 15 | Exact supply and pull-up requirements for each sensor module | Correct wiring |
| 16 | OLED purchase status | Completeness of BOM |
| 17 | Driver-input source for experiments (manual vs scripted) | Repeatable driver behavior |
| 18 | Local ESP32 safety thresholds | Layer-1 protection |
| 19 | Pull-downs on motor-driver lines for boot safety | Prevent motor glitching at boot |
| 20 | Audio sampling rate and window length | Match public datasets and impact detection |
| 21 | Public dataset selection for battery, IMU, road audio | Training data and label compatibility |
| 22 | Temperature sensor placement(s) | Meaningful battery temperature |

---

# 26. Development Roadmap and Review Milestones

## 26.1 Philosophy

```text
UNDERSTAND -> TRAIN -> INTEGRATE -> SIMULATE -> BUILD -> COLLECT -> ADAPT -> VALIDATE -> COMPARE -> PUBLISH
```

Do not try to build everything at once. Make each model work independently, integrate through FastAPI, build the simulation, close the software loop, then build the car, collect data, adapt models, integrate the real-time loop, and compare against the baseline.

## 26.2 Phases

| Phase | Work | Deliverable |
|---|---|---|
| 1 | Literature review (battery SOH/RUL, aging, driving behavior and battery stress, tire-road acoustic sensing, road classification, IMU driving recognition, multimodal fusion, adaptive EV power management, cause analysis) | Literature review, research gap, question, architecture |
| 2 | Public dataset preparation (battery, IMU, road audio): find, download, understand labels, check audio specs, clean, EDA, common pipelines | Datasets + preprocessing |
| 3 | Battery AI model (EDA, features, RF, XGBoost, compare, evaluate, save, inference function) | `battery_model.pkl` |
| 4 | IMU AI model (preprocess, window, filter road vibration, features, train, evaluate incl. vibration robustness) | `imu_model.pkl` |
| 5 | Audio road AI model (preprocess audio, MFCC + spectral features, train RF/XGBoost/SVM, ablation, robustness) | `road_model.pkl` + comparison results |
| 6 | FastAPI backend (app, models, preprocessing, schemas, endpoint, fusion, attribution, protection, Postman tests) | Working AI backend |
| 7 | MATLAB/Simulink (driver, motor, vehicle, road, battery, virtual sensors, scenario controller, FastAPI link, close the loop) | Closed-loop simulation |
| 8 | Dashboard (EV cluster, graphs, simulation control panel) | Dashboard |
| 9 | RC hardware (verification, assembly, power, sensors, mic mounting, PWM, Wi-Fi) | Physical prototype |
| 10 | RC dataset (define experiments, synchronized logging, labeling, audio at multiple speeds, motor-only baselines, repeat across sessions) | DriveGuard-X RC dataset |
| 11 | Model adaptation (evaluate public models on RC data, analyze errors, collect more, retrain, held-out test, before/after) | RC-calibrated models |
| 12 | Final hardware integration (real-time acquisition, inference, fusion, attribution, protection, PWM, OLED/dashboard, end-to-end test) | Real-time DriveGuard-X system |

## 26.3 Review 1 (approximately 30% target)

The first review does not need the physical car complete. The strongest milestone is:

```text
1. Battery AI model        6. Fusion engine
2. IMU AI model            7. Cause attribution
3. Audio road AI model     8. Adaptive protection logic
4. FastAPI backend         9. Digital EV dashboard
5. MATLAB/Simulink sim    10. End-to-end simulation
```

Demonstration:

```text
Simulink -> Virtual sensor data -> FastAPI -> (Battery | IMU | Audio) models
         -> Fusion -> Attribution -> Adaptive protection -> Dashboard
```

Hardware work (verification of BMS, ESP32 bring-up, L298N tests, sensor calibration) can progress in parallel and be shown as a partial hardware status.

## 26.4 After Review 1

```text
Working software system -> RC hardware -> real sensor integration -> RC data collection
 -> target-domain adaptation -> real-time hardware testing -> baseline comparison
 -> final results -> research paper
```

---

# 27. Repository Structure

```text
DriveGuard-X/
|-- README.md
|-- ai/
|   |-- battery/   (notebooks, preprocessing, training, evaluation)
|   |-- imu/       (notebooks, preprocessing, training, evaluation)
|   `-- road/      (audio-only: notebooks, preprocessing, features, training, evaluation)
|-- backend/
|   |-- app/       (main.py, api, models, preprocessing, fusion, attribution, protection)
|   |-- models/
|   `-- requirements.txt
|-- simulation/    (matlab, simulink, scenarios)
|-- hardware/
|   |-- esp32/     (firmware: sensors, comms, safety, motor control)
|   |-- schematics/
|   |-- wiring/    (verified pin-to-pin tables, diagrams)
|   `-- calibration/ (ADC, voltage sensor, ACS712, IMU bias constants)
|-- data/          (public, rc_raw, rc_processed, labels)
|-- dashboard/
|-- docs/          (literature, experiments, architecture, results, test logs)
`-- research/      (figures, tables, paper)
```

---

# 28. Technology Stack

**Hardware:** ESP32 (38-pin), 4 × 18650 (2S2P), 2S balanced BMS, TP5100 charger, 6 V and 5 V buck converters, fuse + main switch, 4 × TT motors (4WD chassis), L298N, ACS712 30 A, 0–25 V voltage sensor, DS18B20, MPU6050, INMP441, SSD1306 OLED.

**AI / data science:** Python, NumPy, Pandas, scikit-learn, XGBoost, SciPy, Librosa (or equivalent), Matplotlib.

**Backend:** FastAPI, Python, Pydantic, REST, JSON.

**Simulation:** MATLAB, Simulink (Simscape / Simscape Electrical / Powertrain Blockset where useful).

**Frontend:** Next.js, React, real-time dashboard.

**Firmware:** Arduino framework or ESP-IDF on ESP32 (FreeRTOS tasks, I²S driver, ADC calibration, PWM via LEDC).

**Communication:** Wi-Fi, HTTP/REST, JSON.

---

# 29. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Unverified cell discharge rating / authenticity | Unsafe current, heating | Verify capacity and datasheet; derate; monitor temperature; keep fuse/BMS |
| Wrong BMS wiring | Pack damage, fire | Identify exact board first; follow manufacturer wiring |
| L298N overload/heat | Driver failure, low motor voltage | Measure current; test per side; consider BTS7960/IBT-2 only if justified |
| L298N logic supply at 6 V | Unreliable logic or back-feed | Resolve logic-rail arrangement before powering |
| ESP32 boot glitches on motor pins | Unexpected motor motion | Pull-downs on driver inputs; wheels lifted for testing |
| ADC noise/nonlinearity, ACS712 offset | Poor battery measurements | Calibrate, average, auto-zero; upgrade only if measurements justify |
| Audio domain shift (full-size vs RC) | Poor road classification | RC data, target-domain adaptation, augmentation |
| Motor noise masks road sound | Audio-only model fails | Mic placement, motor-only baselines, filtering, noise mixing |
| IMU confuses road vibration with aggression | Wrong attribution | Low-pass filtering, low-frequency features, rough-road evaluation |
| Time-series leakage | Inflated results | Run-level and session-level splits |
| Wi-Fi latency/loss | Delayed or missing commands | Command TTL, local limits, safe state |
| Remote-only protection | Unsafe if link fails | Layered safety; ESP32 local limits |
| Over-claiming novelty/causation | Weak research | Position as system-level cause-aware protection; attribution = inference |
| Battery degradation experiments | Safety hazard | Use controlled characterization/public data, not cell abuse |
| Scope creep | Delays | Follow phase order; software-first Review 1 |

---

# 30. Working Rules for Assistants and Collaborators

These rules let this README double as a context document for any assistant or collaborator.

**Role.** Act as technical project assistant, electronics engineer, embedded developer, battery safety advisor, IoT developer, and AI/ML mentor. Provide accurate, practical, beginner-friendly instructions.

**Continuity.**

- Maintain the decisions and components documented here.
- Do not restart the project or change the hardware architecture without explaining why.
- Prefer already-purchased components; recommend replacements only for genuine compatibility, performance, or safety reasons.
- Distinguish **confirmed facts, assumptions, and unresolved specifications**.
- Do not silently treat unresolved specifications as finalized.
- Do not invent specifications. When information is missing, ask for a photograph, product link, or printed model number.
- Do not claim physical testing unless it has occurred.

**When asked for wiring:** provide pin-to-pin tables, explain power and ground, include an overall diagram, identify polarity and voltage hazards, and state which connections need verification against the exact board.

**When asked for component recommendations:** prefer existing components, explain why any replacement is needed, compare specifications rather than trusting titles, search current Indian suppliers if purchase links are requested, and avoid duplicate purchases.

**When asked for code:** use the proposed GPIO map; explain libraries and configuration; separate ESP32 firmware, FastAPI backend, and inference code; include error handling, calibration, and safe motor defaults.

**When advice conflicts with new information:** identify the conflict, explain which specification or measurement resolves it, and correct wrong assumptions rather than preserving them.

**Current stage:** hardware integration. First identify the exact purchased ESP32, L298N, BMS, TP5100, and buck converters (photos, labels, or links), then produce a verified wiring scheme covering the 2S2P battery and BMS, charging circuit, fuse/switch/current sensor, motor and electronics branches, L298N with four TT motors, the ESP32 and all sensors, and the OLED. Do not request new hardware unless a genuine compatibility or safety issue demands it.

**Decisions already made (current direction):**

1. Four-wheel differential-drive chassis.
2. Four yellow TT geared DC motors.
3. ESP32 as main controller.
4. Retain and test the purchased L298N first.
5. 4 × 18650 in 2S2P if the cells pass verification.
6. A correctly rated 2S balanced BMS.
7. One TP5100 configured for 2S, if its exact board/configuration is verified.
8. Regulated 6 V motor supply (not battery direct to TT motors).
9. Separate regulated 5 V electronics supply.
10. ACS712, voltage sensor, DS18B20, MPU6050, INMP441, OLED.
11. Keep the proposed GPIO map unless the actual board requires changes.
12. No ordinary relays as the primary motor driver.
13. No duplicate purchases without checking what is owned.
14. Clear wiring diagrams and pin-to-pin tables.
15. **Road model is audio-only (INMP441); the IMU is for driving behavior only.**

---

# 31. Final Definition of DriveGuard-X

> **A context-aware adaptive battery protection system for electric vehicles that combines battery-state information, driving behavior, and acoustic road-condition sensing to identify the likely cause of battery-stress events and select an appropriate, bounded protection response, validated on an ESP32-controlled 4WD robot-car prototype against a conventional threshold-based baseline.**

```text
Battery sensors (V / I / T)
        +
MPU6050 (driving behavior)
        +
INMP441 (road audio)
        |
        v
Three sensing modalities, three AI models
        |
        v
Multimodal fusion -> Cause attribution -> Adaptive protection
        |
        v
ESP32 local safety limits + bounded PWM -> L298N -> 4 x TT motors
```

**Stage 1 - Software and simulation:** public datasets → AI models → FastAPI → MATLAB/Simulink → dashboard.

**Stage 2 - Hardware and target-domain validation:** 4WD RC car → ESP32 + sensors → real dataset → model adaptation → hardware integration → experimental validation.

The final research question is not simply whether each AI model works. It is whether **context-aware cause attribution can enable more appropriate battery protection than a conventional threshold-based strategy**, while keeping the hardware safe, affordable, and suitable for AI/backend integration.

That is the core of DriveGuard-X.
