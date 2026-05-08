# Hardware bill of materials (draft)

Target budget: roughly 600 to 900 CHF, including buffer for breakage and
iterations. Prices are typical retail figures from 2026; final amounts will
depend on supplier choice.

## Mechanical

| Item | Qty | Notes | Approx. CHF |
|---|---|---|---|
| V-slot 2040 aluminium extrusion, 1.0 m | 1 | Linear rail | 25 |
| V-slot 2040 wheel kit | 1 set | Cart-side rolling element | 35 |
| GT2 timing belt (6 mm, 1.5 m) | 1 | Cart drive | 10 |
| GT2 idler pulley | 2 | Belt return | 10 |
| Carbon-fiber rod, OD 8 mm, 30 cm | 3 | Pendulum links | 30 |
| 608ZZ ball bearings | 6 | Two per hinge | 12 |
| 3D-printed brackets, hinge clamps, end caps | n/a | Assembly hardware | 15 |
| M3/M4 fasteners and shims | 1 set | | 15 |

## Actuator

| Item | Qty | Notes | Approx. CHF |
|---|---|---|---|
| Brushless motor (BLDC, 90 to 150 W) | 1 | Cart actuator | 80 |
| ODrive S1 / Mini | 1 | Brushless driver, current control | 180 |
| Motor mounting bracket | 1 | | 20 |

## Sensing

| Item | Qty | Notes | Approx. CHF |
|---|---|---|---|
| AS5048A magnetic encoder breakout | 3 | 14-bit absolute angle, SPI | 60 |
| Diametric magnet 6 mm | 3 | Pairs with encoder | 6 |
| Linear position sensor or motor encoder | 1 | Cart position; can be the BLDC encoder | included |

## Compute and I/O

| Item | Qty | Notes | Approx. CHF |
|---|---|---|---|
| STM32 Nucleo-F446RE (or similar) | 1 | 1 kHz real-time loop | 25 |
| Raspberry Pi 5 (4 GB) | 1 | Optional bridge to policy host | 80 |
| Logic-level isolation, level shifters | 1 set | Encoder bus protection | 15 |
| 24 V power supply (10 A) | 1 | Motor power | 60 |
| Wiring, connectors, fuses | 1 set | | 30 |

## Subtotals

| Group | Approx. CHF |
|---|---|
| Mechanical | 152 |
| Actuator | 280 |
| Sensing | 66 |
| Compute and I/O | 210 |
| **Subtotal** | **708** |
| 20% contingency | 142 |
| **Estimated total** | **850** |

## Sourcing notes

- AliExpress and Banggood typically halve the price of mechanical extrusions
  and wheels, at the cost of longer lead times and lower QC. Buy spares.
- ODrive is the single biggest line item. A geared brushed motor (JGA25-370)
  with a discrete H-bridge is a viable cheaper alternative if precise current
  control is not required, but it complicates the sim-to-real friction model.
- Carbon-fiber rods can be replaced by PETG-printed tubes for early prototypes;
  carbon is recommended for the final rig because mass and stiffness matter.

## Open issues

- Exact pulley diameter and motor RPM choice (depends on target cart top speed).
- Whether to add a current sensor on the cart drive for sim-to-real torque
  matching.
- Whether to add an emergency end-stop bumper or rely on rail-end detection.
