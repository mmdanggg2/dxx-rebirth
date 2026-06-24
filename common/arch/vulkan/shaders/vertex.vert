#version 450

/* 2D vertex shader. Positions arrive as absolute screen pixel coordinates
 * (origin top-left, +x right, +y down); the push constant carries the
 * pixel->NDC scale/offset so that, combined with the full-screen viewport,
 * a pixel (x,y) lands at framebuffer (x,y).
 *
 *   ndc = pos * scale + offset,  scale=(2/W, 2/H), offset=(-1,-1)
 *
 * Vulkan's viewport (positive height, y=0) maps NDC y=-1 to the top row, so
 * no Y flip is needed in the shader. */
layout(push_constant) uniform pc {
	vec2 scale;
	vec2 offset;
} push;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

void main() {
	gl_Position = vec4(a_pos * push.scale + push.offset, 0.0, 1.0);
	v_uv = a_uv;
	v_color = a_color;
}
