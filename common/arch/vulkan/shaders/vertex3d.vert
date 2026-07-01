#version 450

/* 3D vertex shader. Positions are viewer-relative Descent coordinates
 * (+x right, +y up, +z forward into the screen), converted to float on the
 * CPU via f2fl (fix / 65536). The projection is the 90-degree perspective
 * frustum used by the software renderer and the OpenGL backend
 * (gluPerspective(90,1,0.1,5000)), which reduces to:
 *
 *   ndc.xy = (pos.x / pos.z, -pos.y / pos.z)
 *
 * Vulkan y is downward, so the y term is negated. The depth is the standard
 * hyperbolic near/far mapping, [near,far] -> [0,1]:
 *
 *   depth = far * (z - near) / (z * (far - near))
 *
 * Setting clip-space w = pos.z makes the hardware clip vertices with w <= 0
 * (i.e. z <= 0, behind the viewer) before the perspective divide, giving
 * correct near-plane clipping for free. */

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec3 a_color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

const float near = 0.1;
const float far = 5000.0;

void main() {
	float z = a_pos.z;
	float depth = far * (z - near) / (z * (far - near));
	gl_Position = vec4(a_pos.x, -a_pos.y, depth * z, z);
	v_uv = a_uv;
	v_color = vec4(a_color, 1.0);
}
