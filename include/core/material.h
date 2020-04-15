#include <assimp/material.h>

#include "util.h"

enum BxDFType {
  BSDF_REFLECTION = 1 << 0,
  BSDF_TRANSMISSION = 1 << 1,
  BSDF_DIFFUSE = 1 << 2,
  BSDF_GLOSSY = 1 << 3,
  BSDF_SPECULAR = 1 << 4,
  BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
             BSDF_TRANSMISSION,
};

struct Material {
  Material(aiMaterial *ai_material) {
    aiString ai_name;
    aiColor3D ai_diffuse;
    aiColor3D ai_specular;
    aiColor3D ai_ambient;
    aiColor3D ai_emissive;
    aiColor3D ai_transparent;
    ai_real ai_shininess;
    ai_real ai_shininess_strength;
    ai_real ai_refract;

    ai_material->Get(AI_MATKEY_NAME, ai_name);
    ai_material->Get(AI_MATKEY_COLOR_DIFFUSE, ai_diffuse);
    ai_material->Get(AI_MATKEY_COLOR_SPECULAR, ai_specular);
    ai_material->Get(AI_MATKEY_COLOR_AMBIENT, ai_ambient);
    ai_material->Get(AI_MATKEY_COLOR_EMISSIVE, ai_emissive);
    ai_material->Get(AI_MATKEY_COLOR_TRANSPARENT, ai_transparent);
    ai_material->Get(AI_MATKEY_SHININESS, ai_shininess);
    ai_material->Get(AI_MATKEY_SHININESS_STRENGTH, ai_shininess_strength);
    ai_material->Get(AI_MATKEY_REFRACTI, ai_refract);

    name = std::string(ai_name.C_Str());
    diffuse = Radiance3f(ai_diffuse.r, ai_diffuse.g, ai_diffuse.b);
    specular = Radiance3f(ai_specular.r, ai_specular.g, ai_specular.b);
    ambient = Radiance3f(ai_ambient.r, ai_ambient.g, ai_ambient.b);
    emissive = Radiance3f(ai_emissive.r, ai_emissive.g, ai_emissive.b);
    transparent =
        Radiance3f(ai_transparent.r, ai_transparent.g, ai_transparent.b);
    shininess = ai_shininess;
    shininess_strength = ai_shininess_strength;
    refract = ai_refract;

    bxdf_type = BSDF_DIFFUSE;
  }
  std::string name;  // The name of the material, if available.

  Radiance3f
      diffuse;  // Diffuse color of the material. This is typically scaled
  // by the amount of incoming diffuse light (e.g. using
  // gouraud shading)

  Radiance3f specular;  // Specular color of the material. This is typically
  // scaled by the amount of incoming specular light (e.g.
  // using phong shading)

  Radiance3f
      ambient;  // Ambient color of the material. This is typically scaled
  // by the amount of ambient light

  Radiance3f emissive;  // Emissive color of the material. This is the amount of
  // light emitted by the object. In real time applications
  // it will usually not affect surrounding objects, but
  // raytracing applications may wish to treat emissive
  // objects as light sources.

  Radiance3f
      transparent;  // Defines the transparent color of the material, this
  // is the color to be multiplied with the color of
  // translucent light to construct the final 'destination
  // color' for a particular position in the screen buffer.

  float shininess;  // Defines the shininess of a phong-shaded material. This
  // is actually the exponent of the phong specular equation

  float shininess_strength;  // Scales the specular color of the material.

  float refract;  // Defines the Index Of Refraction for the material. That's
  // not supported by most file formats.

  BxDFType bxdf_type;
};
