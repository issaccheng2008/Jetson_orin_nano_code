# STM32 teaching changes

Repository: `2750257x-blip/Origional_STM32_Control_Code-main`, branch `抬脚`.
Base commit: `246f169c2bff3aed2fb1aa8d5d2f8a31e3f8407a`.

`teaching.patch` contains the teaching and CAN feedback/PD confirmation fixes,
including the CAN transport module and its host tests. Apply to
that baseline using `git apply --check teaching.patch`, then `git apply teaching.patch`.
The complete modified checkout is alongside this repo in `../STM32_reference`.

Build with `cmake --preset Debug` and `cmake --build --preset Debug`.
Prebuilt ELF/BIN outputs are in `../../artifacts/teaching/` (generated, ignored).
Full usage and flashing notes: [teaching console guide](../../docs/teaching_console_zh.md).

The teaching flag is COMMAND bit 3; teaching support is STATE bit 7.
CAN diagnostics capability is STATE bit 8; CAN transport fault is bit 9.
COMMAND remains 152 bytes; the extended STATE is 272 bytes. CRC/version stay v2.
Use the matching Jetson decoder. STM32 branch commit `482535b` includes these fixes.
