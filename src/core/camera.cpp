#include "core/camera.h"

Camera::Camera(const Point3f& position, const Point3f& target,
               const Vector3f& up, const Float focalLength, Float fovy,
               int film_width, int film_height)
    : position(position), up(up), target(target), fovy(fovy) {
  film_size.x = film_width;
  film_size.y = film_height;

  film.resize(static_cast<std::size_t>(film_size.x * film_size.y),
              Vec3f(0.f, 0.f, 0.f));

  Float m = std::min(film_size.x, film_size.y);

  Vector3f cx, cy, cz;

  cz = (target - position).normalize() * -1;
  cx = Cross(up, cz).normalize();
  cy = Cross(cz, cx);

  view_transform = (translate(position.x, position.y, position.z) *
                    Transform(Mat4(cx.x, cy.x, cz.x, 0, cx.y, cy.y, cz.y, 0,
                                   cx.z, cy.z, cz.z, 0, 0, 0, 0, 1)))
                       .inverse();
}

void Camera::setPerspective(Float fovy, Float near, Float far) {
  Float xmin, xmax, ymin, ymax;
  this->fovy = fovy;

  ymax = near * Float(std::tan(fovy * 0.5f));
  ymin = -ymax;
  xmin = ymin / film_size.y * film_size.x;
  xmax = -xmin;

  Float m_array[4][4];

  m_array[0][0] = (2.0f * near) / (xmax - xmin);
  m_array[0][1] = 0.f;
  m_array[0][2] = (xmax + xmin) / (xmax - xmin);
  m_array[0][3] = 0.f;
  m_array[1][0] = 0.f;
  m_array[1][1] = (2.0f * near) / (ymax - ymin);
  m_array[1][2] = (ymax + ymin) / (ymax - ymin);
  m_array[1][3] = 0.f;
  m_array[2][0] = 0.f;
  m_array[2][1] = 0.f;
  m_array[2][2] = -((far + near) / (far - near));
  m_array[2][3] = -((2.0f * far * near) / (far - near));
  m_array[3][0] = 0.f;
  m_array[3][1] = 0.f;
  m_array[3][2] = -1.0f;
  m_array[3][3] = 0.f;

  project_transform = Transform(Mat4(m_array));

  // Transform ndc_to_film = scale(film_size.x, film_size.y, 1.f) *
  //                         translate(Vector3f(0.5f, 0.5f, 0.5f)) *
  //                         scale(0.5, 0.5, 0.5);
  Transform film_to_ndc = translate(Vector3f(-0.5f, -0.5f, -1.f)) *
                          scale(1.0 / film_size.x, 1.0 / film_size.y, 2.f);
  Transform ndc_to_film = film_to_ndc.inverse();
  film_to_camera = (ndc_to_film * project_transform).inverse();

  this->near = near;
  this->far = far;
}

void Camera::writeColor(int r, int c, const Vec3f& color) {
  assert(r >= 0 && r < film_size.y);
  assert(c >= 0 && c < film_size.x);
  film[r * film_size.x + c] = color;
}

Ray Camera::generateRay(const Point2f& pos) {
  // Vector3f&& dir = film_to_camera(Vector3f(pos.x, pos.y, 0)).normalize();
  // Point3f&& origin = Point3f(0, 0, 0) + dir * (near / dir.z);
  // return view_transform.inverse()(Ray(origin, dir));

  const Float aspect = 1.0 * film_size.x / film_size.y;
  const Float s = std::tan(fovy * 0.5f);

  Vector3f&& dir = Vector3f((pos.x / film_size.y - 0.5f * aspect) * s,
                            (pos.y / film_size.y - 0.5f) * s, -1.0f);
  Point3f&& start = Point3f(0, 0, 0) + dir * std::abs(near);

  return view_transform.inverse()(Ray(start, dir.normalize()));
}

Ray Camera::generateRay(const Float x, const Float y) {
  // Vector3f&& dir = film_to_camera(Vector3f(x, y, 0)).normalize();
  // Point3f&& origin = Point3f(0, 0, 0) + dir * (near / dir.z);
  // return view_transform.inverse()(Ray(origin, dir));


  const Float aspect = 1.0 * film_size.x / film_size.y;
  const Float s = std::tan(fovy * 0.5f);

  Vector3f&& dir = Vector3f((x / film_size.y - 0.5f * aspect) * s,
                            (y / film_size.y - 0.5f) * s, -1.0f);
  Point3f&& start = Point3f(0, 0, 0) + dir * std::abs(near);

  return view_transform.inverse()(Ray(start, dir.normalize()));
}