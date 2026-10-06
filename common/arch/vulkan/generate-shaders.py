#!/usr/bin/python
"""Generate a C++ header with embedded SPIR-V shader bytecode."""

import pathlib
import struct
import sys

SPIRV_MAGIC = 0x07230203


def read_spv_words(spv_path: pathlib.Path) -> tuple[int, ...]:
    """Read a SPIR-V module as words, in the byte order its magic number declares."""
    data = spv_path.read_bytes()
    if len(data) % 4:
        raise ValueError(f'{spv_path}: size {len(data)} is not a multiple of 4')
    for order in '<>':
        words = struct.unpack(f'{order}{len(data) // 4}I', data)
        if words and words[0] == SPIRV_MAGIC:
            return words
    raise ValueError(f'{spv_path}: missing SPIR-V magic number')


def format_spv_words(words: tuple[int, ...], width: int = 6) -> str:
    """Format SPIR-V words as C hex literals, wrapping at `width` per line."""
    lines: list[str] = []
    for i in range(0, len(words), width):
        chunk = words[i : i + width]
        lines.append('    ' + ', '.join(f'0x{w:08x}' for w in chunk) + ',')
    return '\n'.join(lines)


def generate_shader_block(name: str, spv_path: pathlib.Path) -> str:
    """Generate the shader block for one SPIR-V file. The words are emitted as
    numbers so the compiler lays them out in the target's byte order, aligned
    as VkShaderModuleCreateInfo::pCode requires."""
    word_array = format_spv_words(read_spv_words(spv_path))
    return f'''
static constexpr uint32_t {name}_spv_words[] = {{
{word_array}
}};

static constexpr const uint32_t *{name}_spv_code() {{
    return {name}_spv_words;
}}

static constexpr std::size_t {name}_spv_size() {{
    return sizeof({name}_spv_words);
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
