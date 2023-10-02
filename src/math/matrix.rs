
use std::ops::{Add, Sub, Mul};
use std::fmt::{self, Formatter, Display};

use super::vector::Vec3;

pub struct Matrix3([Vec3;3]);

impl Matrix3 {
    pub fn new(rows: [Vec3;3]) -> Self {
        Self(rows) 
        
    }

    pub fn zero() -> Self {
        let rows: [Vec3;3] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0)
        ];

        Self(rows)
    
    }

    pub fn identify() -> Self {
        let rows: [Vec3;3] = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0)
        ];

        Self(rows)
    }


    pub fn mul_scale(self, scalar:f32) -> Self {
        let mut res: Self = Self::zero();

        for i in 0..self.0.len(){
            res.0[i] = self.0[i].mul_scale(scalar);
        }


        res
    }

    pub fn rows(self) -> [Vec3;3]{
        self.0
    }

    pub fn transpose(self) -> Self{
        Self(Vec3::transpose(self.0))
    }

    pub fn inverse(self) -> Self{
        Self(Vec3::inverse(self.0))
    }
}

impl Display for Matrix3 {
    // `f` is a buffer, and this method must write the formatted string into it.
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.0.len(){
            write!(f, "({:.4}, {:.4}, {:.4})", self.0[i].x(), self.0[i].y(), self.0[i].z())?;

        }
        write!(f, "]")?;
        Ok(())

    }
}

impl Add for Matrix3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {

        let mut res: Self = Self::zero();
        for i in 0..self.0.len(){
            res.0[i] = self.0[i] + other.0[i];
        }
        res
    }
}

impl Sub for Matrix3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut res: Self = Self::zero();
        for i in 0..self.0.len(){
            res.0[i] = self.0[i] - other.0[i];
        }
        res
    }
}

impl Mul for Matrix3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut res: Self = Self::zero();
        let mut temp0: Self = other.transpose();

        let mut temp: [f32;3] = [0.0, 0.0, 0.0];

        for i in 0..self.0.len(){

            for j in 0..self.0.len(){
                temp[j] = self.0[i].dot(temp0.0[j]);

            }

            res.0[i] = Vec3::new(temp[0], temp[1], temp[2]);
        }

        res
    }

}





