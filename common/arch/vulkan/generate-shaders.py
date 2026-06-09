#!/usr/bin/python
"""Generate a C++ header with embedded SPIR-V shader bytecode."""

import sys
import pathlib


def format_spv_byte(data: bytes, width: int = 12) -> str:
    """Format binary data as C hex-literal byte array, wrapping at `width`."""
    lines: list[str] = []
    for i in range(0, len(data), width):
        chunk = data[i : i + width]
        hex_bytes = ', '.join(f'0x{b:02x}' for b in chunk)
        lines.append(f'    {hex_bytes},')
    return '\n'.join(lines)


def generate_shader_block(name: str, spv_path: pathlib.Path) -> str:
    """Generate the shader block for one SPIR-V file."""
    spv_data = spv_path.read_bytes()
    byte_array = format_spv_byte(spv_data)
    return f'''
static constexpr uint8_t {name}_spv_bytes[] = {{
{byte_array}
}};

static constexpr uint32_t *{name}_spv_code() {{
    return reinterpret_cast<uint32_t *>(const_cast<uint8_t *>({name}_spv_bytes));
}}

static constexpr std::size_t {name}_spv_size() {{
    return sizeof({name}_spv_bytes);
}}
'''


if __name__ == '__main__':
    out_path = pathlib.Path(sys.argv[1])
    spv_paths = sys.argv[2:]

    parts: list[str] = [
        '#pragma once',
        '',
        '/* Auto-generated from SPIR-V shaders — do not edit. */',
        '',
        '#include <cstddef>',
        '#include <cstdint>',
        '',
        'namespace vulkan {',
    ]

    for spv_path_str in spv_paths:
        spv_path = pathlib.Path(spv_path_str)
        name = spv_path.stem  # e.g. "vertex" from "vertex.spv"
        parts.append(generate_shader_block(name, spv_path))

    parts.append('}\n')

    out_path.write_text('\n'.join(parts))
