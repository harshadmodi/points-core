
use std::time::Instant;

use points_core::math::vector::Vec3;

#[test]
fn test_init(){
    let v: Vec3 = Vec3::new(5.0, -9.0, 7.0);
    assert_eq!(v.x(), 5.0);
    assert_eq!(v.y(), -9.0);
    assert_eq!(v.z(), 7.0);
}

#[test]
fn test_dot_product(){
    //let _ = env_logger::builder().is_test(true).try_init();

    let v1: Vec3 = Vec3::new(1.0, 2.0, 3.0);
    let v2: Vec3 = Vec3::new(4.0, -5.0, 6.0);


    let start_time = Instant::now();
    let total:f32 = v1.dot(v2);
    let elapsed_time = start_time.elapsed();
    //info!("Elapsed time: {:?}", elapsed_time);
    assert_eq!(total, 12.0);

}

#[test]
fn test_cross_product(){
    let v1: Vec3 = Vec3::new(2.0, 3.0, 4.0);
    let v2: Vec3 = Vec3::new(5.0, 6.0, 7.0);

    let v3: Vec3;
    v3 = v1.cross(v2);

    assert_eq!(v3.x(), -3.0);
    assert_eq!(v3.y(), 6.0);
    assert_eq!(v3.z(), -3.0);


}

#[test]
fn test_magnitude(){
    let v1: Vec3 = Vec3::new(1.0, 2.0, 3.0);
    let length: f32 = v1.magnitude();
    assert_eq!((length * 1000.0).round() / 1000.0, 3.742 );

}

#[test]
fn test_normalizing(){
    let v1: Vec3 = Vec3::new(1.2, -4.2, 3.5);
    let unit_vector: Vec3 = v1.normalize();

    assert_eq!(unit_vector.x(), 0.2143883);
    assert_eq!(unit_vector.y(), -0.750359);
    assert_eq!(unit_vector.z(), 0.62529916);

}
#[test]
fn test_scalar_projection(){
    let v1: Vec3 = Vec3::new(1.0, 3.0, 7.0);
    let v2: Vec3 = Vec3::new(2.0, 6.0, 3.0);
    let projection: f32 = v1.scalar_projection(v2);
    assert_eq!(projection, 5.857143);
}

#[test]
fn test_vector_projection(){
    let v1: Vec3 = Vec3::new(1.0, 3.0, 7.0);
    let v2: Vec3 = Vec3::new(2.0, 6.0, 3.0);
    let projection: Vec3 = v1.vector_projection(v2);
    assert_eq!(projection.x(), 1.6734694);
    assert_eq!(projection.y(), 5.020408);
    assert_eq!(projection.z(), 2.510204);

}

#[test]
fn test_perpendicular(){
    let v1: Vec3 = Vec3::new(1.0, 3.0, 7.0);
    let v2: Vec3 = Vec3::new(2.0, 6.0, 3.0);
    let perpendicular: Vec3 = v1.perpendicular(v2);
    assert_eq!(perpendicular.x(), -0.6734694);
    assert_eq!(perpendicular.y(), -2.0204082);
    assert_eq!(perpendicular.z(), 4.4897957);

}

#[test]
fn test_orthogonal(){
    let v1: Vec3 = Vec3::new(1.0, 0.0, 1.0);
    let v2: Vec3 = Vec3::new(1.0, 1.0, 0.0);
    let v3: Vec3 = Vec3::new(0.0, 1.0, 1.0);

    let mut e:Vec<Vec3> = Vec::new();
    e.push(v1);
    e.push(v2);
    e.push(v3);
    let orthogonal: Vec<Vec3> = Vec3::orthogonal(e);
    assert_eq!(orthogonal.len(), 3);
    assert_eq!(orthogonal[0].x(), 0.70710677);
    assert_eq!(orthogonal[0].y(), 0.0);
    assert_eq!(orthogonal[0].z(), 0.70710677);
    assert_eq!(orthogonal[1].x(), 0.40824828);
    assert_eq!(orthogonal[1].y(), 0.81649655);
    assert_eq!(orthogonal[1].z(), -0.40824825);
    assert_eq!(orthogonal[2].x(), -0.5773503);
    assert_eq!(orthogonal[2].y(), 0.5773503);
    assert_eq!(orthogonal[2].z(), 0.5773503);
}


