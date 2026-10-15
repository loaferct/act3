# LineGuard pitch: speaker notes

Read these as the narration for the demo video.

## 1. cover

Plants already collect sensor data, but they still break down, because their alarms fire too late to plan a repair. This is LineGuard: it catches machines drifting toward failure about a day earlier than fixed alarms, and an AI copilot running on AMD GPUs turns each alert into a repair plan that can be done in a planned stop.

## 2. problem

Factories already measure vibration, temperature, current and pressure. But their alarms only fire when a fixed limit is crossed. In our evaluation that gave a median of about 10 hours of warning. A bearing kit for a pump takes a day to arrive. So the repair can't be planned, and the machine breaks down during production. In our example plant's history, that cost 11 hours of downtime.

## 3. users

Two people decide whether a fault becomes a breakdown. The shift operator needs to know what is wrong, how urgent it is and what to check. The maintenance planner needs to know which part to order, how long the job takes and which planned stop to use. LineGuard answers both on one screen.

## 4. warning

LineGuard learns what normal looks like for each machine, for every sensor and every hour of the shift. Here, pump P-101's vibration starts to drift away from its normal on Thursday night. The fixed alarm at 7.1 millimetres per second is still far away. LineGuard opens an alert only when the drift holds for 15 minutes, matches the pattern to drive-end bearing wear, and predicts when it will trip by fitting the accelerating trend forward to the trip limit. Here it says about 41 hours; the pump actually tripped 40 hours later.

## 5. plan

One click asks the copilot for a plan. It runs an open model, Qwen3.8-Flash-Next, on AMD GPUs through AMD Token Factory. It reads the alert evidence plus the plant's own manuals, safety procedure and past work orders, and writes a plan an operator can follow: what is happening, what to check, which part, how long, and lockout first. Every line cites its source. The numbers come from LineGuard's code, not from the model.

## 6. loop

The plan becomes a work order with one click. The planner picks a slot, here the next shift change, and approves it. Then we fast-forward to the end of the week. The two machines we acted on were repaired in planned stops. The hydraulic press we ignored broke down. In this one simulated week on one small line, that's 62,750 euros and 11.5 hours of downtime avoided, using the cost assumptions in the repo.

## 7. how

Four steps. Sensor data comes in every minute. The detector compares each sensor with the machine's learned normal, flags persistent drift, matches it to known failure modes and estimates the time to trip. This part is plain code, so it is explainable and testable. The copilot, running on AMD, then finds the right manual sections and past work orders and writes a cited plan. Finally a planner approves a work order into a planned stop.

## 8. amd

The language model runs on AMD GPUs through AMD Token Factory, the hackathon's AI API. We use Qwen3.8-Flash-Next, an open-weights model. A full plan takes 18 to 30 seconds at about 20 to 25 tokens per second, and in our benchmark 9 out of 9 plans were valid and cited. In the demo it feels instant, because the analysis starts in the background as soon as an alert opens. The same code also runs against a vLLM server on an AMD MI300X, so a plant can keep its data on site.

## 9. results

We measured this on 50 random fault scenarios the detector never saw, 149 developing faults in total. LineGuard warned a median of 33 hours before the trip, against 10 hours for fixed alarms. It caught every fault before the trip and raised no false alerts on healthy machines. Under our explicit cost assumptions, that's about 18,700 euros and 3 hours of downtime saved per fault. To be clear: the plant data is simulated, so real data will be noisier, and anyone can rerun this with one command.

## 10. trust

In a factory, trust matters more than cleverness. Every statement the copilot makes cites a source, and citations it can't back up are struck out. Lockout/tagout is enforced for any hands-on work. The numbers come from code, not the model. A person approves every work order. If the AI is down, a labelled fallback keeps the screen working. And tests cover the whole loop.

## 11. business

A plant pays because each caught fault avoids roughly 18,700 euros under our example cost model, and LineGuard uses sensors the plant already has. We propose 150 euros per monitored machine per month, 1,800 euros a year, so a single caught fault a year pays for about ten machines. Next steps: connect to a real plant historian or OPC UA, pilot on one line with real failure history, self-host the model on an AMD MI300X so data stays on site, and learn new failure modes from the plant's own work orders.

## 12. close

LineGuard: catch it a day earlier, fix it in a planned stop. The code and the demo video are linked here. Thank you.
