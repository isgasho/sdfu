use std::ops::*;

/// Functionality that should be shared between all vector types.
pub trait Vec<T>:
    Sized
    + Copy
    + Neg<Output = Self>
    + Mul<T, Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<T, Output = Self>
    + Div<T, Output = Self>
    + MaxMin
    + Zero
    + One
    + Clamp
{
    type Dimension: Dimension;
    type Vec2: Vec2<T>;
    type Vec3: Vec3<T>;
    fn dot(&self, other: Self) -> T;
    fn magnitude(&self) -> T;
    fn magnitude_squared(&self) -> T;
    fn abs(&self) -> Self;
    fn normalized(&self) -> Self;
    fn euc_mod(&self, other: Self) -> Self;
    fn round(&self) -> Self;
}

/// Functionality that must be implmeented by 3D vectors.
pub trait Vec3<T>: Vec<T> {
    fn new(x: T, y: T, z: T) -> Self;
    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
}

/// Functionality that must be implmeented by 2D vectors.
pub trait Vec2<T>: Vec<T> {
    fn new(x: T, y: T) -> Self;
    fn x(&self) -> T;
    fn y(&self) -> T;
}

/// A trait used to mark the dimensionality of a vector/SDF/implmentation
/// of an SDF combinator.
pub trait Dimension {}

/// 2D marker struct.
#[derive(Clone, Copy, Debug)]
pub struct Dim2D {}

/// 2D marker struct.
#[derive(Clone, Copy, Debug)]
pub struct Dim3D {}

impl Dimension for Dim2D {}
impl Dimension for Dim3D {}

/// Return the maximum or minimum of `self` and `other`.
pub trait MaxMin {
    #[inline(always)]
    fn max(&self, other: Self) -> Self;
    #[inline(always)]
    fn min(&self, other: Self) -> Self;
}

impl MaxMin for f32 {
    fn max(&self, other: Self) -> Self {
        f32::max(*self, other)
    }
    fn min(&self, other: Self) -> Self {
        f32::min(*self, other)
    }
}

impl MaxMin for f64 {
    fn max(&self, other: Self) -> Self {
        f64::max(*self, other)
    }
    fn min(&self, other: Self) -> Self {
        f64::min(*self, other)
    }
}

/// The additive identity.
pub trait Zero {
    #[inline(always)]
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

/// The multiplicative identity.
pub trait One {
    #[inline(always)]
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

/// Linear interpolate between self and other with a factor
/// between Self::zero() and Self::one.
pub trait Lerp {
    fn lerp(&self, other: Self, factor: Self) -> Self;
}

impl<T> Lerp for T
where
    T: Copy + Mul<T, Output = T> + Sub<T, Output = T> + Add<T, Output = T> + One,
{
    fn lerp(&self, other: Self, factor: Self) -> Self {
        *self * (T::one() - factor) + other * factor
    }
}

/// Clamp the value(s) of self to between `low` and `high`.
pub trait Clamp {
    fn clamp(&self, low: Self, high: Self) -> Self;
}

/// Raises `2^(self)`
pub trait Exp2 {
    #[inline(always)]
    fn exp2(&self) -> Self;
}

impl Exp2 for f32 {
    fn exp2(&self) -> Self {
        f32::exp2(*self)
    }
}

impl Exp2 for f64 {
    fn exp2(&self) -> Self {
        f64::exp2(*self)
    }
}

/// Returns log base 2 of self.
pub trait Log2 {
    #[inline(always)]
    fn log2(&self) -> Self;
}

impl Log2 for f32 {
    fn log2(&self) -> Self {
        f32::log2(*self)
    }
}

impl Log2 for f64 {
    fn log2(&self) -> Self {
        f64::log2(*self)
    }
}

/// This is a trait for types that can rotate an SDF.
/// Note that the implementation should actually rotate the vec
/// by the *inverse* of the native rotation that the type would
/// normally produce.
pub trait Rotation<V> {
    fn rotate_vec(&self, v: V) -> V;
}

#[cfg(feature = "vek")]
pub mod vek_integration {
    use super::*;

    impl<T: vek::ops::Clamp + Copy> Clamp for T {
        fn clamp(&self, low: T, high: T) -> T {
            self.clamped(low, high)
        }
    }

    impl<T: Zero> Zero for vek::vec::Vec2<T> {
        fn zero() -> Self {
            vek::vec::Vec2::new(T::zero(), T::zero())
        }
    }

    impl<T: Zero> Zero for vek::vec::Vec3<T> {
        fn zero() -> Self {
            vek::vec::Vec3::new(T::zero(), T::zero(), T::zero())
        }
    }

    impl<T: One> One for vek::vec::Vec2<T> {
        fn one() -> Self {
            vek::vec::Vec2::new(T::one(), T::one())
        }
    }

    impl<T: One> One for vek::vec::Vec3<T> {
        fn one() -> Self {
            vek::vec::Vec3::new(T::one(), T::one(), T::one())
        }
    }

    impl<T: PartialOrd + Copy> MaxMin for vek::vec::Vec2<T> {
        fn max(&self, other: Self) -> Self {
            vek::vec::Vec2::partial_max(*self, other)
        }

        fn min(&self, other: Self) -> Self {
            vek::vec::Vec2::partial_min(*self, other)
        }
    }

    impl<T: PartialOrd + Copy> MaxMin for vek::vec::Vec3<T> {
        fn max(&self, other: Self) -> Self {
            vek::vec::Vec3::partial_max(*self, other)
        }

        fn min(&self, other: Self) -> Self {
            vek::vec::Vec3::partial_min(*self, other)
        }
    }

    macro_rules! impl_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec2<$inner_t> for vek::vec::Vec2<$inner_t> {
                fn new(x: $inner_t, y: $inner_t) -> Self {
                    vek::vec::Vec2::new(x, y)
                }
                fn x(&self) -> $inner_t { self.x }
                fn y(&self) -> $inner_t { self.y }
            })+
        }
    }
    impl_vec2!(f32, f64);

    macro_rules! impl_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec3<$inner_t> for vek::vec::Vec3<$inner_t> {
                fn new(x: $inner_t, y: $inner_t, z: $inner_t) -> Self {
                    vek::vec::Vec3::new(x, y, z)
                }
                fn x(&self) -> $inner_t { self.x }
                fn y(&self) -> $inner_t { self.y }
                fn z(&self) -> $inner_t { self.z }
            })+
        }
    }
    impl_vec3!(f32, f64);

    macro_rules! impl_vec_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for vek::vec::Vec2<$inner_t>
            {
                type Dimension = Dim2D;
                type Vec2 = Self;
                type Vec3 = vek::vec::Vec3<$inner_t>;
                fn dot(&self, other: Self) -> $inner_t {
                    vek::vec::Vec2::dot(*self, other)
                }

                fn magnitude(&self) -> $inner_t {
                    vek::vec::Vec2::magnitude(*self)
                }

                fn magnitude_squared(&self) -> $inner_t {
                    vek::vec::Vec2::magnitude_squared(*self)
                }

                fn abs(&self) -> Self {
                    vek::vec::Vec2::new(self.x.abs(), self.y.abs())
                }

                fn normalized(&self) -> Self {
                    vek::vec::Vec2::normalized(*self)
                }

                fn euc_mod(&self, other: Self) -> Self {
                    vek::vec::Vec2::new(self.x.rem_euclid(other.x), self.y.rem_euclid(other.y))
                }

                fn round(&self) -> Self {
                    vek::vec::Vec2::new(self.x.round(), self.y.round())
                }
            })+
        }
    }
    impl_vec_vec2!(f32, f64);

    macro_rules! impl_vec_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for vek::vec::Vec3<$inner_t>
            {
                type Dimension = Dim3D;
                type Vec2 = vek::vec::Vec2<$inner_t>;
                type Vec3 = Self;
                fn dot(&self, other: Self) -> $inner_t {
                    vek::vec::Vec3::dot(*self, other)
                }

                fn magnitude(&self) -> $inner_t {
                    vek::vec::Vec3::magnitude(*self)
                }

                fn magnitude_squared(&self) -> $inner_t {
                    vek::vec::Vec3::magnitude_squared(*self)
                }

                fn abs(&self) -> Self {
                    vek::vec::Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
                }

                fn normalized(&self) -> Self {
                    vek::vec::Vec3::normalized(*self)
                }

                fn euc_mod(&self, other: Self) -> Self {
                    vek::vec::Vec3::new(self.x.rem_euclid(other.x), self.y.rem_euclid(other.y), self.z.rem_euclid(other.z))
                }

                fn round(&self) -> Self {
                    vek::vec::Vec3::new(self.x.round(), self.y.round(), self.z.round())
                }
            })+
        }
    }
    impl_vec_vec3!(f32, f64);

    macro_rules! impl_rotation_mat {
        {$($outer_t:ty => $inner_t:ty),+} => {
            $(impl Rotation<$inner_t> for $outer_t {
                fn rotate_vec(&self, v: $inner_t) -> $inner_t {
                    <$inner_t>::from(vek::mat::Mat4::from(*self).inverted() * vek::vec::Vec4::from(v))
                }
            })+
        }
    }

    impl_rotation_mat! {
        vek::mat::Mat3<f32> => vek::vec::Vec2<f32>,
        vek::mat::Mat2<f32> => vek::vec::Vec2<f32>,
        vek::mat::Mat3<f32> => vek::vec::Vec3<f32>,
        vek::mat::Mat4<f32> => vek::vec::Vec3<f32>,
        vek::mat::Mat3<f64> => vek::vec::Vec2<f64>,
        vek::mat::Mat2<f64> => vek::vec::Vec2<f64>,
        vek::mat::Mat3<f64> => vek::vec::Vec3<f64>,
        vek::mat::Mat4<f64> => vek::vec::Vec3<f64>
    }

    macro_rules! impl_rotation_quat {
        {$($outer_t:ty => $inner_t:ty),+} => {
            $(impl Rotation<$inner_t> for $outer_t {
                fn rotate_vec(&self, v: $inner_t) -> $inner_t {
                    <$inner_t>::from(self.inverse() * vek::vec::Vec4::from(v))
                }
            })+
        }
    }

    impl_rotation_quat! {
        vek::quaternion::Quaternion<f32> => vek::vec::Vec2<f32>,
        vek::quaternion::Quaternion<f32> => vek::vec::Vec3<f32>,
        vek::quaternion::Quaternion<f64> => vek::vec::Vec2<f64>,
        vek::quaternion::Quaternion<f64> => vek::vec::Vec3<f64>
    }
}

#[cfg(not(feature = "vek"))]
impl Clamp for f32 {
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }
}

#[cfg(not(feature = "vek"))]
impl Clamp for f64 {
    fn clamp(&self, low: Self, high: Self) -> Self {
        self.max(low).min(high)
    }
}

#[cfg(feature = "nalgebra")]
pub mod nalgebra_integration {
    use super::*;
    use nalgebra as na;
    use std::fmt::Debug;

    impl<T: PartialOrd + Copy + Debug + 'static> Clamp for na::Vector2<T> {
        fn clamp(&self, low: Self, high: Self) -> Self {
            na::Vector2::new(
                na::clamp(self.x, low.x, high.x),
                na::clamp(self.y, low.y, high.y),
            )
        }
    }

    impl<T: PartialOrd + Copy + Debug + 'static> Clamp for na::Vector3<T> {
        fn clamp(&self, low: Self, high: Self) -> Self {
            na::Vector3::new(
                na::clamp(self.x, low.x, high.x),
                na::clamp(self.y, low.y, high.y),
                na::clamp(self.z, low.z, high.z),
            )
        }
    }

    impl<T: Zero + PartialEq + Copy + Debug + 'static> Zero for na::Vector2<T> {
        fn zero() -> Self {
            na::Vector2::new(T::zero(), T::zero())
        }
    }

    impl<T: Zero + PartialEq + Copy + Debug + 'static> Zero for na::Vector3<T> {
        fn zero() -> Self {
            na::Vector3::new(T::zero(), T::zero(), T::zero())
        }
    }

    impl<T: One + PartialEq + Copy + Debug + 'static> One for na::Vector2<T> {
        fn one() -> Self {
            na::Vector2::new(T::one(), T::one())
        }
    }

    impl<T: One + PartialEq + Copy + Debug + 'static> One for na::Vector3<T> {
        fn one() -> Self {
            na::Vector3::new(T::one(), T::one(), T::one())
        }
    }

    impl<T: PartialOrd + Copy + Debug + 'static> MaxMin for na::Vector2<T> {
        fn max(&self, other: Self) -> Self {
            na::Vector2::new(
                *na::partial_max(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_max(&self.y, &other.y).unwrap_or(&self.y),
            )
        }

        fn min(&self, other: Self) -> Self {
            na::Vector2::new(
                *na::partial_min(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_min(&self.y, &other.y).unwrap_or(&self.y),
            )
        }
    }

    impl<T: PartialOrd + Copy + Debug + 'static> MaxMin for na::Vector3<T> {
        fn max(&self, other: Self) -> Self {
            na::Vector3::new(
                *na::partial_max(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_max(&self.y, &other.y).unwrap_or(&self.y),
                *na::partial_max(&self.z, &other.z).unwrap_or(&self.z),
            )
        }

        fn min(&self, other: Self) -> Self {
            na::Vector3::new(
                *na::partial_min(&self.x, &other.x).unwrap_or(&self.x),
                *na::partial_min(&self.y, &other.y).unwrap_or(&self.y),
                *na::partial_min(&self.z, &other.z).unwrap_or(&self.z),
            )
        }
    }

    macro_rules! impl_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec2<$inner_t> for na::Vector2<$inner_t> {
                fn new(x: $inner_t, y: $inner_t) -> Self {
                    na::Vector2::new(x, y)
                }
                fn x(&self) -> $inner_t { self.x }
                fn y(&self) -> $inner_t { self.y }
            })+
        }
    }
    impl_vec2!(f32, f64);

    macro_rules! impl_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec3<$inner_t> for na::Vector3<$inner_t> {
                fn new(x: $inner_t, y: $inner_t, z: $inner_t) -> Self {
                    na::Vector3::new(x, y, z)
                }
                fn x(&self) -> $inner_t { self.x }
                fn y(&self) -> $inner_t { self.y }
                fn z(&self) -> $inner_t { self.z }
            })+
        }
    }
    impl_vec3!(f32, f64);

    macro_rules! impl_vec_vec2 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for na::Vector2<$inner_t>
            {
                type Dimension = Dim2D;
                type Vec2 = Self;
                type Vec3 = na::Vector3<$inner_t>;
                fn dot(&self, other: Self) -> $inner_t {
                    na::Vector2::dot(self, &other)
                }

                fn magnitude(&self) -> $inner_t {
                    na::Vector2::magnitude(self)
                }

                fn magnitude_squared(&self) -> $inner_t {
                    na::Vector2::magnitude_squared(self)
                }

                fn abs(&self) -> Self {
                    na::Vector2::new(self.x.abs(), self.y.abs())
                }

                fn normalized(&self) -> Self {
                    na::Vector2::normalize(self)
                }

                fn euc_mod(&self, other: Self) -> Self {
                    na::Vector2::new(self.x.rem_euclid(other.x), self.y.rem_euclid(other.y))
                }

                fn round(&self) -> Self {
                    na::Vector2::new(self.x.round(), self.y.round())
                }
            })+
        }
    }
    impl_vec_vec2!(f32, f64);

    macro_rules! impl_vec_vec3 {
        ($($inner_t:ty),+) => {
            $(impl Vec<$inner_t> for na::Vector3<$inner_t>
            {
                type Dimension = Dim3D;
                type Vec2 = na::Vector2<$inner_t>;
                type Vec3 = Self;
                fn dot(&self, other: Self) -> $inner_t {
                    na::Vector3::dot(self, &other)
                }

                fn magnitude(&self) -> $inner_t {
                    na::Vector3::magnitude(self)
                }

                fn magnitude_squared(&self) -> $inner_t {
                    na::Vector2::magnitude_squared(self)
                }

                fn abs(&self) -> Self {
                    na::Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
                }

                fn normalized(&self) -> Self {
                    na::Vector3::normalize(self)
                }

                fn euc_mod(&self, other: Self) -> Self {
                    na::Vector3::new(self.x.rem_euclid(other.x), self.y.rem_euclid(other.y), self.z.rem_euclid(other.z))
                }

                fn round(&self) -> Self {
                    na::Vector3::new(self.x.round(), self.y.round(), self.z.round())
                }
            })+
        }
    }
    impl_vec_vec3!(f32, f64);

    macro_rules! impl_rot_inner {
        ($($rot_ty:ty => $vec_ty:ty),+) => {
            $(impl Rotation<$vec_ty> for $rot_ty {
                fn rotate_vec(&self, v: $vec_ty) -> $vec_ty {
                    self.inverse_transform_vector(&v)
                }
            })+
        }
    }

    macro_rules! impl_rot {
        ($($inner_ty:ty),+) => {
            $(impl_rot_inner!(
                na::Rotation2<$inner_ty> => na::Vector2<$inner_ty>,
                na::UnitComplex<$inner_ty> => na::Vector2<$inner_ty>,
                na::Rotation3<$inner_ty> => na::Vector3<$inner_ty>,
                na::UnitQuaternion<$inner_ty> => na::Vector3<$inner_ty>
            );)+
        }
    }
    impl_rot!(f32, f64);
}
