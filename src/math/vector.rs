//! # Vec3
//!
//! `Vec3` is a collection of Vector Operations
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::fmt::{self, Formatter, Display};

#[derive(Clone, Copy, Debug)]
pub struct Vec3 (__m128);

macro_rules! _mm_shuffle {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

impl Display for Vec3 {
    // `f` is a buffer, and this method must write the formatted string into it.
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        
        write!(f, "({:.4}, {:.4}, {:.4})", self.x(), self.y(), self.z())
    }
}


impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        unsafe { 
            Vec3(_mm_set_ps(0.0, z, y, x)) 
        }
    }
    /// Calculate dot product.
    pub fn dot(self, v: Self) -> f32 {
        //  Example :
        //  Vector a (x=1.0, y=2.0, z=3.0, w=0)
        //  Vector b (x=4.0, y=-5.0, z=6.0, w=0)

        //  c = [ax*bx, ay*by, az*bz, aw*bw] = [1*4, -2*5, 3*6, 0*0] = [4, -10, 18, 0]

        //  temp1 = c + c = [4, -10, 18, 0] + [4, -10, 18, 0] = [-6, 18, -6, 18]
        //  temp2 = temp1 + temp1 = [-6 18, -6, 18] + [-6, 18, -6, 18] = [12, 12, 12, 12]
        // --snip--
        let mut total:f32 = 0.0;
        let mut num: __m128;
        unsafe {
            num = _mm_mul_ps(self.0, v.0); //performs multiplication   num = a[3]*b[3]  a[2]*b[2]  a[1]*b[1]  a[0]*b[0]
            num = _mm_hadd_ps(num, num); // performs horizontal addition num =  a[3]*b[3]+ a[2]*b[2]  a[1]*b[1]+a[0]*b[0]  a[3]*b[3]+ a[2]*b[2]  a[1]*b[1]+a[0]*b[0]
            num = _mm_hadd_ps(num, num); // performs horizontal addition
            total = _mm_cvtss_f32(num);
        }
        return total;

    }

    pub fn x(self) -> f32 {
        unsafe { _mm_cvtss_f32(self.0) }
    }

    pub fn y(self) -> f32 {
        /*

        m3 = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(z, y, x, w)               [ mask: (z << 6) | (y << 4) | (x << 2) | w]
        m3 = (m2(z) << 6) | (m2(y) << 4) | (m1(x) << 2) | m1(w).

        Example: 
         [4, 2, 3, 1] => [2, 3, 1, 4].

         +-----------+---------+
        | index     | 3 2 1 0 |
        +-----------+---------+
        | element   | 4 2 3 1 |
        +-----------+---------+

        
        
         We need to pass the initial pack, [4, 2, 3, 1] twice: 
         
         _mm_shuffle_ps([4, 2, 3, 1], [4, 2, 3, 1], mask) 

         and form a mask, which will use elements [2, 3] for the higher words of a result and elements [3, 1] for the lower words. 
         These elements can be then indexed as follows:

         So to get the pair [2, 3] we need elements with indices [2, 1]. 
         And to get the pair [1, 4] we need elements with indices [0, 3].

        Given that, we can use macro _MM_SHUFFLE() to generate the mask: _MM_SHUFFLE(2, 1, 0, 3). 
        And the final formula looks like this: _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(2, 1, 0, 3)).

        ref: https://shybovycha.github.io/2017/02/21/speeding-up-algorithms-with-sse.html


        Example:
        Vector (x=5, y=-9, z=7, w=0) convert into Vector (x=-9, y=-9, z=-9, w=-9)
         
        [0, 7, -9 , 5] => [-9, -9, -9, -9]
         +-----------+---------+
        | index     | 3 2 1 0 |
        +-----------+----------+
        | element   | 0 7 -9 5 |
        +-----------+---------+

        mask = [1, 1, 1, 1]

        */
        
        unsafe { _mm_cvtss_f32(_mm_shuffle_ps(self.0, self.0, _mm_shuffle!(1, 1, 1, 1))) }
    }

    pub fn z(self) -> f32 {
        /*
        Example:
        Vector (x=5, y=-9, z=7, w=0) convert into Vector (x=7, y=7, z=7, w=7)
         
        [0, 7, -9 , 5] => [7, 7, 7, 7]

         +-----------+---------+
        | index     | 3 2 1 0 |
        +-----------+----------+
        | element   | 0 7 -9 5 |
        +-----------+---------+

        mask = [2, 2, 2, 2]
        */
        unsafe { _mm_cvtss_f32(_mm_shuffle_ps(self.0, self.0, _mm_shuffle!(2, 2, 2, 2))) }
    }

    pub fn cross(self, v: Self) -> Self {
        /*
        Formula :

        |a.x|   |b.x|   | a.y * b.z - a.z * b.y |
        |a.y| X |b.y| = | a.z * b.x - a.x * b.z |
        |a.z|   |b.z|   | a.x * b.y - a.y * b.x |

        Example :
         Vector a (x=2.0, y=3.0, z=4.0, w=0)
          +-----------+---------+
        | index     | 3 2 1 0 |
        +-----------+----------+      
        | element   | 0 4 3 2 |
        +-----------+---------+
        | temp0     | 0 2 4 3 |      mask: (3, 0, 2, 1)     
        +-----------+----------+
        | temp2     | 0 3 2 4 |      mask: (3, 1, 0, 2)
        +-----------+---------+

         Vector b (x=5.0, y=6.0, z=7.0, w=0)

          +-----------+---------+
        | index     | 3 2 1 0 |
        +-----------+----------+
        | element   | 0 7 6 5 |
        +-----------+---------+
        | temp1     | 0 6 5 7 |      mask: (3, 1, 0, 2)     
        +-----------+----------+
        | temp3     | 0 5 7 6 |      mask: (3, 0, 2, 1)
        +-----------+---------+

        c1 = temp0 * temp1 = [0, 2, 4, 3] * [0, 6, 5, 7] = [0, 12, 20, 21]
        c2 = temp2 * temmp3 = [0, 3, 2, 4] * [0, 5, 7, 6] = [0, 15, 14, 24]

        c = c1 - c2 = [0, 12, 20, 21] - [0, 15, 14, 24] = [0, -3, 6, -3]

        */
        let tmp0: __m128;
        let tmp1: __m128;
        let tmp2: __m128;
        let tmp3: __m128;
        let res: __m128;

        unsafe {
            
            tmp0 = _mm_shuffle_ps(self.0, self.0, _mm_shuffle!(3, 0, 2, 1));
            tmp1 = _mm_shuffle_ps(v.0, v.0, _mm_shuffle!(3, 1, 0, 2));

            tmp2 = _mm_shuffle_ps(self.0, self.0, _mm_shuffle!(3, 1, 0, 2));
            tmp3 = _mm_shuffle_ps(v.0, v.0, _mm_shuffle!(3, 0, 2, 1));

            res = _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
        }
        Self(res)
    }

    pub fn magnitude(self) -> f32 {
        //  Example : | (1, 2, 3)T |   =  √( 12 + 22 + 32 )   =  √( 1 + 4 + 9 )  =  √14   =   3.742 
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        /* Example 
        

            The dot product is:

                ( 1.2, -4.2, 3.5 )T · ( 1.2, -4.2, 3.5 )T   =   1.22 + (-4.2)2 + 3.52   =   31.33. 

            The unit vector is represented by:

                ( 1.2, -4.2, 3.5 )T / √31.33.   =   (0.2144, -0.7504, 0.6253)T 

        */
        let inv_length = 1.0 / self.magnitude();
        unsafe { 
            Self(_mm_mul_ps(self.0, _mm_set1_ps(inv_length))) 
        }
    }

    pub fn scalar_projection(self, v:Self) -> f32 {
        /*
        a = (1, 3, 7)
        b = (2, 6, 3)
        Scalar Projection of a vector on b vector  =  (a . b) / |b|

        a vector . b vector  =  1(2) + 3(6) + 7(3) =  2 + 18 + 21 =  41
        |b vector|  =  √(22 + 62 + 32)   =  √49  =  7

        Scalar Projection of a vector on b vector  =  41/7

        ref: https://flexbooks.ck12.org/cbook/ck-12-college-precalculus/section/9.6/primary/lesson/scalar-and-vector-projections-c-precalc/
        */
        return 1.0 / v.magnitude() * self.dot(v);

    }

    pub fn vector_projection(self, v:Self) -> Self {
        /*
        Vector Projection of a vector onto b vector = ( (a . b) / |b| ) *  ( b / |b| )

        a = (1, 3, 7)
        b = (2, 6, 3)

        Vector Projection of a vector onto b vector  =  41/7 * (2/7, 6/7, 3/7) = (82/49, 246/49, 123/7)
                                                     = (1.673469, 5.020408, 2.510204)



        */
        let scalar_projection:f32 = self.scalar_projection(v);
        let scalar:f32 = scalar_projection / v.magnitude();
        let mut num: __m128;

        let mut scalar: Self = Self::new(scalar, scalar, scalar);
        unsafe {
            num = _mm_mul_ps(v.0, scalar.0);
        }
        Self(num)


    }

    pub fn perpendicular(self, v:Vec3) -> Vec3 {
        /*
        the component of a that is perpendicular to b = a - vector Projection of a vector onto b vector
                                                      = (1, 3, 7) - (1.673469, 5.020408, 2.510204)
                                                      = (-0.673469, −2.020408, 4.489796)


        */
        let mut num: __m128;
        let mut projection: Self = self.vector_projection(v);
        unsafe {
            num = _mm_sub_ps(self.0, projection.0);
        }
        Self(num)
    }

    pub fn mul_scale(self, scalar:f32) -> Self {

        let mut num: __m128;

        let mut scalar: Self = Self::new(scalar, scalar, scalar);
        unsafe {
            num = _mm_mul_ps(self.0, scalar.0);
        }

        Self(num)

    }

    pub fn orthogonal(vectors:Vec<Self>) -> Vec<Self> {
        /* 
        The orthogonal vectors produced by Gram-Schmidt can be written in terms of projectors

        Classical Gram-Schmidt

        for j=1:n
            Vj = Xj
            for k=1:j−1
                Vj = vj − (QkTXj)Qk
            endfor
            Qj = Vj/∥Vj∥2
        endfor

        ref: https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html


        
        */
        let mut U :Vec<Vec3> = Vec::new();
        let vec_len = vectors.len();

        let mut Q :Vec<Vec3> = Vec::new();

        for i in 0..vec_len {
            U.push(vectors[i]);

            let x = vectors[i];
            
            for j in 0..i {
                let q = Q[j];
                
                let r:Self = q.mul_scale(q.dot(x));

                let mut num: __m128;
                
                unsafe {
                    num = _mm_sub_ps(U[i].0, r.0);
                }

                U[i] = Self(num);
            }

            let scalar:f32 = 1.0 / U[i].magnitude();
            let r:Self = U[i].mul_scale(scalar);

            Q.push(r);
        }
        Q
    }
    pub fn modified_orthogonal(vectors:Vec<Self>) -> Vec<Self> {
        /*
        Modified Gram-Schmidt Algorithm
        
        for i=1 to n
            Ui = Vi
        for i=1 to n
            Qi = Ui * 1 / || Ui ||
            for j=i+1 to n
                r = Qi dot Uj
                Uj = Uj - ( Qi * r )
                
        
        Example
        a1 = [1, 0, 1] 
        a2 = [1, 1, 0]
        a3 = [0, 1, 1]

        u1 = [1, 0, 1]  ||u1|| = √2
        q1 = [1, 0, 1] * √2 = [1/√2 , 0, 1/√2] = (0.7071, 0.0000, 0.7071)

        r12 = q1 dot u2 * q1 = 1 / √2 (1/√2, 0, 1/√2) = (0.5, 0, 0.5)
        u2 = u2 - r12 = (1, 1, 0) - 1 / √2 (1/√2, 0, 1/√2) = (1/2, 1, -1/2) = (0.5, 1, -0.5)
        r13 = q1 dot u3 * q1 = (0.5, 0.0, 0.5)
        u3 = u3 - r13 = (0, 1, 1) - (0.5, 0.0, 0.5) = (-0.5, 1, 0.5)

        q2 = [0.5, 1, -0.5] * 0.8165 = (0.4082, 0.8165, -0.4082)

        r23 = (0.1667, 0.3333, -0.1667)
        u3 = (-0.6667, 0.6667, 0.6667)

        q3 = (-0.5774, 0.5774, 0.5774)

        */

        let vec_len = vectors.len();

        let mut U :Vec<Vec3> = Vec::new();
        let mut Q :Vec<Vec3> = Vec::new();

        for i in 0..vec_len {
            U.push(vectors[i]);
        }
        for i in 0..vec_len {
            let u = U[i];
            
            let scalar:f32 = 1.0 / u.magnitude();
            let r:Self = u.mul_scale(scalar);

            Q.push(r);

            for j in i+1..vec_len {
                let q = Q[i];
                let u = U[j];
                let r:Self = q.mul_scale(q.dot(u));

                let mut num: __m128;

                unsafe {
                    num = _mm_sub_ps(u.0, r.0);
                }

                U[j] = Self(num);
            }
        }

        Q

    }

    
}






