#include "transform.h"

Transform::Transform() {}

Transform::Transform(Mat4 m)
{
    mat = m;
    inv = m.inverse();
}

Transform::Transform(Mat4 m, Mat4 inv)
{
    this->mat = m;
    this->inv = inv;
}

Transform Transform::getMatrix() const
{
    return this->mat;
}

Transform Transform::getInverseMatrix() const
{
    return this->inv;
}

Transform Transform::inverse() const
{
    return Transform(this->inv, this->mat);
}

Transform Transform::transpose() const
{
    return Transform(this->mat.transpose(), this->inv.transpose());
}

inline Point3f Transform::operator()(const Point3f &p) const
{
    float xx = mat.m[0][0] * p.x + mat.m[0][1] * p.y + mat.m[0][2] * p.z + mat.m[0][3];
    float yy = mat.m[1][0] * p.x + mat.m[1][1] * p.y + mat.m[1][2] * p.z + mat.m[1][3];
    float zz = mat.m[2][0] * p.x + mat.m[2][1] * p.y + mat.m[2][2] * p.z + mat.m[2][3];
    float ww = mat.m[3][0] * p.x + mat.m[3][1] * p.y + mat.m[3][2] * p.z + mat.m[3][3];
    if (ww == 1)
        return Point3f(xx, yy, zz);
    else
        return Point3f(xx / ww, yy / ww, zz / ww);
}

inline Vector3f Transform::operator()(const Vector3f &v) const
{
    return Vector3f(mat.m[0][0] * v.x + mat.m[0][1] * v.y + mat.m[0][2] * v.z,
                    mat.m[1][0] * v.x + mat.m[1][1] * v.y + mat.m[1][2] * v.z,
                    mat.m[2][0] * v.x + mat.m[2][1] * v.y + mat.m[3][2] * v.z);
}

inline Normal3f Transform::operator()(const Normal3f &n) const
{
    return Normal3f(inv.m[0][0] * n.x + inv.m[1][0] * n.y + inv.m[2][0] * n.z,
                    inv.m[0][1] * n.x + inv.m[1][1] * n.y + inv.m[2][1] * n.z,
                    inv.m[0][2] * n.x + inv.m[1][2] * n.y + inv.m[2][2] * n.z);
}
inline Ray Transform::operator()(const Ray &r)const{

}

Transform translate(const Vector3f &delta)
{
    Mat4 mat(1, 0, 0, delta.x,
             0, 1, 0, delta.y,
             0, 0, 1, delta.z,
             0, 0, 0, 1);
    Mat4 inv(1, 0, 0, -delta.x,
             0, 1, 0, -delta.y,
             0, 0, 1, -delta.z,
             0, 0, 0, 1);
    return Transform(mat, inv);
}

Transform scale(float x, float y, float z)
{
    Mat4 mat(x, 0, 0, 0,
             0, y, 0, 0,
             0, 0, 0, z,
             0, 0, 0, 1);

    Mat4 inv(1 / x, 0, 0, 0,
             0, 1 / y, 0, 0,
             0, 0, 1 / z, 0,
             0, 0, 0, 1);

    return Transform(mat, inv);
}

Transform rotateX(float theta)
{
    float sinTheta = std::sinf(theta);
    float cosTheta = std::cosf(theta);
    Mat4 mat(1, 0, 0, 0,
             0, cosTheta, -sinTheta, 0,
             0, sinTheta, cosTheta, 0,
             0, 0, 0, 1);

    Mat4 inv(1, 0, 0, 0,
             0, cosTheta, sinTheta, 0,
             0, -sinTheta, cosTheta, 0,
             0, 0, 0, 1);
    return Transform(mat, inv);
}

Transform rotateY(float theta)
{
    float sinTheta = std::sinf(theta);
    float cosTheta = std::cosf(theta);
    Mat4 mat(cosTheta, 0, sinTheta, 0,
             0, 1, 0, 0,
             -sinTheta, 0, cosTheta, 0,
             0, 0, 0, 1);

    Mat4 inv(cosTheta, 0, -sinTheta, 0,
             0, 1, 0, 0,
             sinTheta, 0, cosTheta, 0,
             0, 0, 0, 1);
    return Transform(mat, inv);
}

Transform rotateZ(float theta)
{
    float sinTheta = std::sinf(theta);
    float cosTheta = std::cosf(theta);
    Mat4 mat(cosTheta, -sinTheta, 0, 0,
             sinTheta, cosTheta, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1);

    Mat4 inv(cosTheta, sinTheta, 0, 0,
             -sinTheta, cosTheta, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1);
}

Transform rotate(float theta, const Vector3f &axis)
{
    Vector3f a = axis.normalize();
    float sinTheta = std::sinf(theta);
    float cosTheta = std::cosf(theta);
    Mat4 m;
    // Compute rotation of first basis vector
    m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m.m[0][3] = 0;

    // Compute rotations of second and third basis vectors
    m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m.m[1][3] = 0;

    m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m.m[2][3] = 0;
    return Transform(m, m.transpose());
}

Transform lookAt(const Point3f &pos, const Point3f &look, const Vector3f &up)
{
    Mat4 camera2World;
    camera2World.m[0][3] = pos.x;
    camera2World.m[1][3] = pos.y;
    camera2World.m[2][3] = pos.z;
    camera2World.m[3][3] = 1;

    Vector3f dir = (look - pos).normalize();
    Vector3f right = cross(up.normalize(), dir).normalize();
    Vector3f newUp = cross(dir, right);

    camera2World.m[0][0] = right.x;
    camera2World.m[1][0] = right.y;
    camera2World.m[2][0] = right.z;
    camera2World.m[3][0] = 0.;
    camera2World.m[0][1] = newUp.x;
    camera2World.m[1][1] = newUp.y;
    camera2World.m[2][1] = newUp.z;
    camera2World.m[3][1] = 0.;
    camera2World.m[0][2] = dir.x;
    camera2World.m[1][2] = dir.y;
    camera2World.m[2][2] = dir.z;
    camera2World.m[3][2] = 0.;
    return Transform(camera2World.inverse(), camera2World);
}
