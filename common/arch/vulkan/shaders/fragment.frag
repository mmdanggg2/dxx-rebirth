#version 450

/* 2D fragment shader. Samples the bound texture and modulates by the
 * per-vertex color. For flat primitives (rects/lines/pixels) a 1x1 white
 * texture is bound, so the result is just the vertex color.
 *
 * Color-key transparency: transparent palette entries are uploaded with
 * alpha 0; discarding them yields crisp edges. */
layout(set = 0, binding = 0) uniform sampler2D s_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main() {
	vec4 c = texture(s_tex, v_uv);
	if (c.a == 0.0)
		discard;
	out_color = c * v_color;
}
