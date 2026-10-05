#version 450

/* Shared fragment shader. Samples the bound texture and modulates by the
 * per-vertex color. For flat primitives (rects/lines/pixels) a 1x1 white
 * texture is bound, so the result is just the vertex color.
 *
 * Alpha test: transparent palette entries are uploaded with alpha 0, and
 * like the OpenGL backend's glAlphaFunc(GL_GEQUAL, 0.02) fragments whose
 * modulated alpha (texture * vertex fade) falls below 0.02 are discarded. */
layout(set = 0, binding = 0) uniform sampler2D s_tex;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main() {
	vec4 c = texture(s_tex, v_uv) * v_color;
	if (c.a < 0.02)
		discard;
	out_color = c;
}
