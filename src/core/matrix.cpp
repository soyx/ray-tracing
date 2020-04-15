#include "core/matrix.h"

Mat4::Mat4()
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (i == j)
                m[i][j] = 1.0f;
            else
                m[i][j] = 0.0f;
}

Mat4::Mat4(const Float mat[4][4])
{
    std::memcpy(m, mat, 16 * sizeof(Float));
}

Mat4::Mat4(Float m00, Float m01, Float m02, Float m03,
           Float m10, Float m11, Float m12, Float m13,
           Float m20, Float m21, Float m22, Float m23,
           Float m30, Float m31, Float m32, Float m33)
{
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[0][3] = m03;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[1][3] = m13;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
    m[2][3] = m23;
    m[3][0] = m30;
    m[3][1] = m31;
    m[3][2] = m32;
    m[3][3] = m33;
}

Mat4 Mat4::Identity(){
    return Mat4();
}

Mat4 Mat4::operator=(const Mat4 &m2)
{
    std::memcpy(m, m2.m, 16 * sizeof(Float));
    return *this;
}

Mat4 Mat4::operator+(const Mat4 &m2) const
{
    Float mtemp[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            mtemp[i][j] = m[i][j] + m2.m[i][j];
    }
    return Mat4(mtemp);
}

Mat4 Mat4::operator-(const Mat4 &m2) const
{
    Float mtemp[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            mtemp[i][j] = m[i][j] - m2.m[i][j];
    }
    return Mat4(mtemp);
}

Mat4 Mat4::operator*(const Mat4 &m2) const
{
    Mat4 ans;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ans.m[i][j] = m[i][0] * m2.m[0][j] + m[i][1] * m2.m[1][j] +
                          m[i][2] * m2.m[2][j] + m[i][3] * m2.m[3][j];
    return ans;
}

bool Mat4::operator==(const Mat4 &m2) const
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (m[i][j] != m2.m[i][j])
                return false;

    return true;
}

bool Mat4::operator!=(const Mat4 &m2) const
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (m[i][j] != m2.m[i][j])
                return true;

    return false;
}

Mat4 Mat4::inverse()const
{
    int indxc[4], indxr[4];
    int ipiv[4] = {0, 0, 0, 0};
    Float minv[4][4];
    memcpy(minv, this->m, 16 * sizeof(Float));
    for (int i = 0; i < 4; i++)
    {
        int irow = 0, icol = 0;
        Float big = 0.f;
        // Choose pivot
        for (int j = 0; j < 4; j++)
        {
            if (ipiv[j] != 1)
            {
                for (int k = 0; k < 4; k++)
                {
                    if (ipiv[k] == 0)
                    {
                        if (std::fabs(minv[j][k]) >= big)
                        {
                            big = (Float)std::fabs(minv[j][k]);
                            irow = j;
                            icol = k;
                        }
                    }
                    else if (ipiv[k] > 1)
                        std::cout << "ERROR: Singular matrix in MatrixInvert" << std::endl;
                }
            }
        }
        ++ipiv[icol];
        if (irow != icol)
        {
            for (int k = 0; k < 4; k++)
                std::swap(minv[irow][k], minv[icol][k]);
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (minv[icol][icol] == 0.f)
            std::cout << "ERROR: Singualr matrix in MatrixInvert" << std::endl;

        Float pivinv = 1. / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < 4; j++)
            minv[icol][j] *= pivinv;

        for (int j = 0; j < 4; j++)
        {
            if (j != icol)
            {
                Float save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++)
                    minv[j][k] -= minv[icol][k] * save;
            }
        }
    }

    for (int j = 3; j >= 0; j--)
    {
        if (indxr[j] != indxc[j])
        {
            for (int k = 0; k < 4; k++)
            {
                std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
            }
        }
    }

    return Mat4(minv);
}

Mat4 Mat4::transpose() const
{
    return Mat4(m[0][0], m[1][0], m[2][0], m[3][0],
                m[0][1], m[1][1], m[2][1], m[3][1],
                m[0][2], m[1][2], m[2][2], m[3][2],
                m[0][3], m[1][3], m[2][3], m[3][3]);
}