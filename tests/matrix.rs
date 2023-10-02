use points_core::math::matrix::Matrix3;
use points_core::math::vector::Vec3;


#[test]
fn test_matrix_transpose(){

    let r1: Vec3 = Vec3::new(1.0, 2.0, -3.0);
    let r2: Vec3 = Vec3::new(4.0, -5.0, 6.0);
    let r3: Vec3 = Vec3::new(7.0, 8.0, -9.0);

    let m1:Matrix3 = Matrix3::new([r1, r2, r3]);

    let t:Matrix3 = m1.transpose();

    let rows = t.rows();

    assert_eq!(rows[0].x(), 1.0);
    assert_eq!(rows[0].y(), 4.0);
    assert_eq!(rows[0].z(), 7.0);

    assert_eq!(rows[1].x(), 2.0);
    assert_eq!(rows[1].y(), -5.0);
    assert_eq!(rows[1].z(), 8.0);

    assert_eq!(rows[2].x(), -3.0);
    assert_eq!(rows[2].y(), 6.0);
    assert_eq!(rows[2].z(), -9.0);


}

#[test]
fn test_matrix_inverse(){

    let r1: Vec3 = Vec3::new(4.0, 2.0, -3.0);
    let r2: Vec3 = Vec3::new(1.0, -1.0, 2.0);
    let r3: Vec3 = Vec3::new(5.0, 3.0, 0.0);

    let m1:Matrix3 = Matrix3::new([r1, r2, r3]);

    let i:Matrix3 = m1.inverse();

    let rows = i.rows();

    assert_eq!(rows[0].x(), 0.21428573);
    assert_eq!(rows[0].y(), 0.3214286);
    assert_eq!(rows[0].z(), -0.035714286);

    assert_eq!(rows[1].x(), -0.357142857);
    assert_eq!(rows[1].y(), -0.5357143);
    assert_eq!(rows[1].z(), 0.39285716);

    assert_eq!(rows[2].x(), -0.285714286);
    assert_eq!(rows[2].y(), 0.071428571);
    assert_eq!(rows[2].z(), 0.21428573);


}

#[test]
fn test_matrix_add(){

    let r1: Vec3 = Vec3::new(3.0, -2.0, 6.0);
    let r2: Vec3 = Vec3::new(1.0, 7.0, 9.0);
    let r3: Vec3 = Vec3::new(-3.0, 4.0, 2.0);

    let m1:Matrix3 = Matrix3::new([r1, r2, r3]);


    let r4: Vec3 = Vec3::new(8.0, 1.0, 3.0);
    let r5: Vec3 = Vec3::new(-6.0, 1.0, 1.0);
    let r6: Vec3 = Vec3::new(2.0, 1.0, 7.0);

    let m2:Matrix3 = Matrix3::new([r4, r5, r6]);


    let m3:Matrix3 = m1 + m2;

    let rows = m3.rows();

    assert_eq!(rows[0].x(), 11.0);
    assert_eq!(rows[0].y(), -1.0);
    assert_eq!(rows[0].z(), 9.0);

    assert_eq!(rows[1].x(), -5.0);
    assert_eq!(rows[1].y(), 8.0);
    assert_eq!(rows[1].z(), 10.0);

    assert_eq!(rows[2].x(), -1.0);
    assert_eq!(rows[2].y(), 5.0);
    assert_eq!(rows[2].z(), 9.0);


    
}
#[test]
fn test_matrix_substraction(){

    let r1: Vec3 = Vec3::new(2.0, 1.0, 0.0);
    let r2: Vec3 = Vec3::new(5.0, 6.0, 0.0);
    let r3: Vec3 = Vec3::new(2.0, 9.0, 0.0);

    let m1:Matrix3 = Matrix3::new([r1, r2, r3]);


    let r4: Vec3 = Vec3::new(3.0, 6.0, 0.0);
    let r5: Vec3 = Vec3::new(8.0, 1.0, 0.0);
    let r6: Vec3 = Vec3::new(7.0, 0.0, 0.0);

    let m2:Matrix3 = Matrix3::new([r4, r5, r6]);


    let m3:Matrix3 = m1 - m2;

    let rows = m3.rows();


    assert_eq!(rows[0].x(), -1.0);
    assert_eq!(rows[0].y(), -5.0);
    assert_eq!(rows[0].z(), 0.0);

    assert_eq!(rows[1].x(), -3.0);
    assert_eq!(rows[1].y(), 5.0);
    assert_eq!(rows[1].z(), 0.0);

    assert_eq!(rows[2].x(), -5.0);
    assert_eq!(rows[2].y(), 9.0);
    assert_eq!(rows[2].z(), 0.0);


    
}

#[test]
fn test_matrix_multiplication(){

    let r1: Vec3 = Vec3::new(2.0, 3.0, 1.0);
    let r2: Vec3 = Vec3::new(7.0, 4.0, 1.0);
    let r3: Vec3 = Vec3::new(9.0, -2.0, 1.0);

    let m1:Matrix3 = Matrix3::new([r1, r2, r3]);


    let r4: Vec3 = Vec3::new(9.0, -2.0, -1.0);
    let r5: Vec3 = Vec3::new(5.0, 7.0, 3.0);
    let r6: Vec3 = Vec3::new(8.0, 1.0, 0.0);

    let m2:Matrix3 = Matrix3::new([r4, r5, r6]);


    let m3:Matrix3 = m1 * m2;

    let rows = m3.rows();

    assert_eq!(rows[0].x(), 41.0);
    assert_eq!(rows[0].y(), 18.0);
    assert_eq!(rows[0].z(), 7.0);

    assert_eq!(rows[1].x(), 91.0);
    assert_eq!(rows[1].y(), 15.0);
    assert_eq!(rows[1].z(), 5.0);

    assert_eq!(rows[2].x(), 79.0);
    assert_eq!(rows[2].y(), -31.0);
    assert_eq!(rows[2].z(), -15.0);

    
}

#[test]
fn test_matrix_scale(){

    let r1: Vec3 = Vec3::new(1.0, 2.0, 3.0);
    let r2: Vec3 = Vec3::new(4.0, -5.0, 6.0);
    let r3: Vec3 = Vec3::new(7.0, 8.0, 9.0);

    let m1:Matrix3 = Matrix3::new([r1, r2, r3]);
    
    let m3:Matrix3 = m1.mul_scale(2.0);

    let rows = m3.rows();

    assert_eq!(rows[0].x(), 2.0);
    assert_eq!(rows[0].y(), 4.0);
    assert_eq!(rows[0].z(), 6.0);

    assert_eq!(rows[1].x(), 8.0);
    assert_eq!(rows[1].y(), -10.0);
    assert_eq!(rows[1].z(), 12.0);

    assert_eq!(rows[2].x(), 14.0);
    assert_eq!(rows[2].y(), 16.0);
    assert_eq!(rows[2].z(), 18.0);


    
}

