#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;


macro_rules! _mm_shuffle {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

macro_rules! _mm_shuffle_mask {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        $x | ($y << 2) | ($z << 4) | ($w << 6)
    };
    
}


macro_rules! _vec_swizzle_mask {
    ($vec:expr, $mask:expr) => {
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128($vec), $mask))
    };
}

macro_rules! _vec_swizzle {
    ($vec:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {
        crate::math::helper::_vec_swizzle_mask!($vec, crate::math::helper::_mm_shuffle_mask!($x, $y, $z, $w))
    };
}


macro_rules! _vec_swizzle1 {
    ($vec:expr, $x:expr) => {
        crate::math::helper::_vec_swizzle_mask!($vec, crate::math::helper::_mm_shuffle_mask!($x, $x, $x, $x))
    };
}

macro_rules! _vec_swizzle_0022 {
    ($vec:expr) => {
        _mm_moveldup_ps($vec)
    };
}


macro_rules! _vec_swizzle_1133 {
    ($vec:expr) => {
        _mm_movehdup_ps($vec)
    };
}


macro_rules! _vec_shuffle {
    ($vec1:expr, $vec2:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {
        _mm_shuffle_ps($vec1, $vec2, crate::math::helper::_mm_shuffle_mask!($x, $y, $z, $w))
    };
}

macro_rules! _vec_shuffle_0101 {
    ($vec1:expr, $vec2:expr) => {
        _mm_movelh_ps($vec1, $vec2)
    };
}


macro_rules! _vec_shuffle_2323 {
    ($vec1:expr, $vec2:expr) => {
        _mm_movehl_ps($vec2, $vec1)
    };
}

pub(crate) use _mm_shuffle;
pub(crate) use _mm_shuffle_mask;

pub(crate) use _vec_shuffle_0101;
pub(crate) use _vec_shuffle_2323;
pub(crate) use _vec_swizzle_mask;

pub(crate) use _vec_shuffle;

pub(crate) use _vec_swizzle1;
pub(crate) use _vec_swizzle;


pub fn mat2mul(vec1:__m128, vec2:__m128) -> __m128{
	unsafe {
		_mm_add_ps(
            _mm_mul_ps(vec1, 
                _vec_swizzle!(vec2,0,0,3,3)
            ),
		    _mm_mul_ps(
                _vec_swizzle!(vec1,2,3,0,1), 
                _vec_swizzle!(vec2,1,1,2,2)
            )
        )
    }
}

pub fn mat2adjmul(vec1:__m128, vec2:__m128) -> __m128 {
	unsafe {
		_mm_sub_ps(
            _mm_mul_ps(
                _vec_swizzle!(vec1, 3,0,3,0), 
                vec2
            ),
		    _mm_mul_ps(
                _vec_swizzle!(vec1, 2,1,2,1), 
                _vec_swizzle!(vec2, 1,0,3,2)
            )
        )
    }

}

pub fn mat2muladj(vec1:__m128, vec2:__m128) -> __m128 {
	unsafe {
		_mm_sub_ps(
            _mm_mul_ps(vec1, 
                _vec_swizzle!(vec2, 3,3,0,0)
            ),
		    _mm_mul_ps(
                _vec_swizzle!(vec1, 2,3,0,1), 
                _vec_swizzle!(vec2, 1,1,2,2)
            )
        )
    }
}