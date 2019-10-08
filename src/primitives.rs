//! A collection of primitive SDFs that may the modified using functions in `mods`
//! or combined using functions in `ops`. Note that these primitives are always
//! centered around the origin and that you must transform the point you are sampling
//! into 'primitive-local' space. Functions are provided in `mods` to do this easier.
//!
//! Also note that while all translation and rotation transformations of the input point
//! will work properly, scaling modifies the Euclidian space and therefore does not work
//! normally. Utility function shave been provided to `Translate`, `Rotate`, and `Scale`
//! in the `SDF` trait and the `mods` module.
use super::*;
use crate::mathtypes::*;
use std::marker::PhantomData;

macro_rules! impl_primitives {
    ($($t:ty),+) => {
        /// An octahedron centered at origin.
        #[derive(Clone, Copy, Debug)]
        pub struct Octahedron {}

        impl Octahedron {
            pub fn new() -> Self {
                Octahedron {}
            }
        }

        $(impl<V> Sdf<$t, V> for Octahedron
        where
            V: Vec3<$t>,
        {
            fn dist(&self, p: V) -> $t {
                let p = p.abs();
                let m = p.x()+p.y()+p.z()-1.0;
                let q = if 3.0*p.x() < m {
                    p
                } else if 3.0*p.y() < m {
                    V::new(p.y(), p.z(), p.x())
                } else if 3.0*p.z() < m {
                    V::new(p.z(), p.x(), p.y())
                } else {
                    return m*0.57735027
                };

                let k = Clamp::clamp(&(0.5*(q.z()-q.y()+1.0)), 0.0, 1.0);
                V::new(q.x(), q.y()- 1.0 +k ,q.z()-k).magnitude()
            }
        })+

        /// A shere centered at origin with a radius.
        #[derive(Clone, Copy, Debug)]
        pub struct Sphere<T> {
            pub radius: T,
        }

        $(impl Sphere<$t> {
            pub fn new(radius: $t) -> Self {
                Sphere { radius }
            }
        }

        impl<V> Sdf<$t, V> for Sphere<$t>
        where
            V: Vec3<$t>,
        {
            fn dist(&self, p: V) -> $t {
                p.magnitude() - self.radius
            }
        })+

        /// A box centered at origin with axis-aligned dimensions.
        #[derive(Clone, Copy, Debug)]
        pub struct Box<V, D> {
            pub dims: V,
            _pd: PhantomData<D>,
        }

        impl<V, D> Box<V, D> {
            pub fn new(dims: V) -> Self {
                Box {
                    dims,
                    _pd: PhantomData,
                }
            }
        }

        $(impl<V> Sdf<$t, V> for Box<V, Dim3D>
        where
            V: Vec3<$t> + Copy,
        {
            fn dist(&self, p: V) -> $t {
                let d = p.abs() - self.dims;
                d.max(V::zero()).magnitude() + d.y().max(d.z()).max(d.x()).min(0.0)
            }
        }

        impl<V> Sdf<$t, V> for Box<V, Dim2D>
        where
            V: Vec2<$t> + Copy,
        {
            fn dist(&self, p: V) -> $t {
                let d = p.abs() - self.dims;
                d.max(V::zero()).magnitude() + d.y().max(d.x()).min(0.0)
            }
        })+

        /// A circle centered at origin with a radius.
        #[derive(Clone, Copy, Debug)]
        pub struct Circle<T> {
            pub radius: T,
        }

        $(impl Circle<$t> {
            pub fn new(radius: $t) -> Self {
                Circle { radius }
            }
        }

        impl<V> Sdf<$t, V> for Circle<$t>
        where
            V: Vec2<$t>,
        {
            fn dist(&self, p: V) -> $t {
                p.magnitude() - self.radius
            }
        })+

        /// A torus that sits on the XZ plane. Thickness is the radius of
        /// the wrapped cylinder while radius is the radius of the donut
        /// shape.
        #[derive(Clone, Copy, Debug)]
        pub struct Torus<T> {
            pub radius: T,
            pub thickness: T,
        }

        $(impl Torus<$t> {
            pub fn new(radius: $t, thickness: $t) -> Self {
                Torus { radius, thickness }
            }
        }

        impl<V> Sdf<$t, V> for Torus<$t>
        where
            V: Vec3<$t>,
        {
            fn dist(&self, p: V) -> $t {
                let q = V::Vec2::new(
                    V::Vec2::new(p.x(), p.z()).magnitude() - self.thickness,
                    p.y(),
                );
                q.magnitude() - self.radius
            }
        })+

        #[derive(Clone, Copy, Debug)]
        pub enum Axis {
            X,
            Y,
            Z,
        }

        /// An infinite cylinder extending along an axis.
        #[derive(Clone, Copy, Debug)]
        pub struct Cylinder<T> {
            pub radius: T,
            pub axis: Axis,
        }

        $(impl Cylinder<$t> {
            pub fn new(radius: $t, axis: Axis) -> Self {
                Cylinder { radius, axis }
            }
        }

        impl<V> Sdf<$t, V> for Cylinder<$t>
        where
            V: Vec3<$t>,
        {
            fn dist(&self, p: V) -> $t {
                let (a, b) = match self.axis {
                    Axis::X => (p.y(), p.z()),
                    Axis::Y => (p.x(), p.z()),
                    Axis::Z => (p.x(), p.y()),
                };
                V::Vec2::new(a, b).magnitude() - self.radius
            }
        })+

        /// A capped cylinder extending along an axis.
        #[derive(Clone, Copy, Debug)]
        pub struct CappedCylinder<T> {
            pub radius: T,
            pub height: T,
            pub axis: Axis,
        }

        $(impl CappedCylinder<$t> {
            pub fn new(radius: $t, height: $t, axis: Axis) -> Self {
                CappedCylinder {
                    radius,
                    height,
                    axis,
                }
            }
        }

        impl<V> Sdf<$t, V> for CappedCylinder<$t>
        where
            V: Vec3<$t>,
        {
            fn dist(&self, p: V) -> $t {
                let (a, b, c) = match self.axis {
                    Axis::X => (p.y(), p.z(), p.x()),
                    Axis::Y => (p.x(), p.z(), p.y()),
                    Axis::Z => (p.x(), p.y(), p.z()),
                };
                let d = V::Vec2::new(V::Vec2::new(a, b).magnitude(), c).abs()
                    - V::Vec2::new(self.radius, self.height);
                d.x().max(d.y()).min(0.0) + d.max(V::Vec2::zero()).magnitude()
            }
        })+

        /// A capsule extending from `a` to `b` with thickness `thickness`.
        #[derive(Clone, Copy, Debug)]
        pub struct Line<T, V> {
            pub a: V,
            pub b: V,
            pub thickness: T,
        }

        $(impl<V> Line<$t, V> {
            pub fn new(a: V, b: V, thickness: $t) -> Self {
                Line { a, b, thickness }
            }
        }

        impl<V> Sdf<$t, V> for Line<$t, V>
        where
            V: Vec<$t> + Copy,
        {
            fn dist(&self, p: V) -> $t {
                let pa = p - self.a;
                let ba = self.b - self.a;
                let t = pa.dot(ba) / ba.dot(ba);
                let h = Clamp::clamp(&t, 0.0, 1.0);
                (pa - (ba * h)).magnitude() - self.thickness
            }
        })+
    };
}

impl_primitives!(f32, f64);
