# STM32 teaching changes

Repository: `2750257x-blip/Origional_STM32_Control_Code-main`, branch `抬脚`.
Base commit: `246f169c2bff3aed2fb1aa8d5d2f8a31e3f8407a`.

`teaching.patch` contains the four firmware source/header changes. Apply to
that baseline using `git apply --check teaching.patch`, then `git apply teaching.patch`.
The complete modified checkout is alongside this repo in `../STM32_reference`.

Build with `cmake --preset Debug` and `cmake --build --preset Debug`.
Prebuilt ELF/BIN outputs are in `../../artifacts/teaching/` (generated, ignored).
Full usage and flashing notes: [teaching console guide](../../docs/teaching_console_zh.md).

The new teaching flag is COMMAND bit 3; firmware support is STATE bit 7.
Message layouts and CRC remain protocol v2. Firmware changes are required:
the console refuses motor enable if the capability bit is absent.
