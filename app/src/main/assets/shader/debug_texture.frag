#version 320 es
precision mediump float;

in vec2 v_texCoords;

out vec4 frag_color;

uniform sampler2D screen_shot;

void main()
{
    vec4 color = texture(screen_shot, v_texCoords);

    frag_color = color;
}