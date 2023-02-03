use std::{marker::PhantomData};

use group::ff::Field;
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector, Constraints, Expression},
    poly::Rotation,
};
use std::{any::Any};
use halo2_proofs::{pasta::Fp};

/// A variable representing a number.
#[derive(Clone)]
struct Number<F: Field>(AssignedCell<F, F>);

// ANCHOR: add-instructions
trait AddInstructions<F: Field>: Chip<F> {
    /// Variable representing a number.
    type Num;

    /// Returns `c = a + b`.
    fn add(
        &self,
        layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error>;
}
// ANCHOR_END: add-instructions

// ANCHOR: add-config
#[derive(Clone, Debug)]
struct AddConfig {
    advice: [Column<Advice>; 2],
    s_add: Selector,
}
// ANCHOR_END: add-config

// ANCHOR: add-chip
struct AddChip<F: Field> {
    config: AddConfig,
    _marker: PhantomData<F>,
}
// ANCHOR END: add-chip

// ANCHOR: add-chip-trait-impl
impl<F: Field> Chip<F> for AddChip<F> {
    type Config = AddConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}
// ANCHOR END: add-chip-trait-impl

// ANCHOR: add-chip-impl
impl<F: Field> AddChip<F> {
    fn construct(config: <Self as Chip<F>>::Config, _loaded: <Self as Chip<F>>::Loaded) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 2],
    ) -> <Self as Chip<F>>::Config {
        let s_add = meta.selector();

        // Define our addition gate!
        meta.create_gate("add", |meta| {
            let lhs = meta.query_advice(advice[0], Rotation::cur());
            let rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[0], Rotation::next());
            let s_add = meta.query_selector(s_add);

            vec![s_add * (lhs + rhs - out)]
        });

        AddConfig { advice, s_add }
    }
}
// ANCHOR END: add-chip-impl

// ANCHOR: add-instructions-impl
impl<F: Field> AddInstructions<F> for FieldChip<F> {
    type Num = Number<F>;
    fn add(
        &self,
        layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error> {
        let config = self.config().add_config.clone();

        let add_chip = AddChip::<F>::construct(config, ());
        add_chip.add(layouter, a, b)
    }
}

impl<F: Field> AddInstructions<F> for AddChip<F> {
    type Num = Number<F>;

    fn add(
        &self,
        mut layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error> {
        let config = self.config();

        layouter.assign_region(
            || "add",
            |mut region: Region<'_, F>| {
                // We only want to use a single addition gate in this region,
                // so we enable it at region offset 0; this means it will constrain
                // cells at offsets 0 and 1.
                config.s_add.enable(&mut region, 0)?;

                // The inputs we've been given could be located anywhere in the circuit,
                // but we can only rely on relative offsets inside this region. So we
                // assign new cells inside the region and constrain them to have the
                // same values as the inputs.
                a.0.copy_advice(|| "lhs", &mut region, config.advice[0], 0)?;
                b.0.copy_advice(|| "rhs", &mut region, config.advice[1], 0)?;

                // Now we can compute the addition result, which is to be assigned
                // into the output position.
                let value = a.0.value().copied() + b.0.value();

                // Finally, we do the assignment to the output, returning a
                // variable to be used in another part of the circuit.
                region
                    .assign_advice(|| "lhs + rhs", config.advice[0], 1, || value)
                    .map(Number)
            },
        )
    }
}
// ANCHOR END: add-instructions-impl

// ANCHOR: relu-instructions
trait ReluInstructions<F: Field>: Chip<F> {
    /// Variable representing a number.
    type Num;

    /// Returns `c = clipped((input * a) / b, 0, 255)`.
    fn relu(
        &self,
        layouter: impl Layouter<F>,
        input: Self::Num,
        a: Self::Num,
        b: usize,
    ) -> Result<Self::Num, Error>;
}
// ANCHOR_END: relu-instructions

// ANCHOR: relu-config
#[derive(Clone, Debug)]
struct ReluConfig {
    advice: [Column<Advice>; 2],
    s_relu: Selector,
}
// ANCHOR_END: relu-config

// ANCHOR: relu-chip
struct ReluChip<F: Field> {
    config: ReluConfig,
    _marker: PhantomData<F>,
}
// ANCHOR END: relu-chip

// ANCHOR: relu-chip-trait-impl
impl<F: Field> Chip<F> for ReluChip<F> {
    type Config = ReluConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}
// ANCHOR END: relu-chip-trait-impl

// ANCHOR: relu-chip-impl
impl<F: Field> ReluChip<F> {
    fn construct(config: <Self as Chip<F>>::Config, _loaded: <Self as Chip<F>>::Loaded) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }
    
    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 2],
    ) -> <Self as Chip<F>>::Config {
        let s_relu = meta.selector();
        // Define our reluition gate!
        meta.create_gate("relu", |meta| {
            let _lhs = meta.query_advice(advice[0], Rotation::cur());
            let _rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[0], Rotation::next());
            let s_relu = meta.query_selector(s_relu);

            let range_check = |range: usize, value: Expression<F>| {
                assert!(range > 0);
                (1..range).fold(value.clone(), |expr, i| {
                    expr * (Expression::Constant(*as_any_gen(&Fp::from(i as u64)).downcast_ref::<F>().expect("Error!")) - value.clone())
                })
            };
            Constraints::with_selector(s_relu.clone(), [("range check", range_check(256, out.clone()))])
            // return vec![s_relu * (out.clone() - out)];
        });

        ReluConfig { advice, s_relu }
    }
}
// ANCHOR END: relu-chip-impl

// ANCHOR: relu-instructions-impl
impl<F: Field> ReluInstructions<F> for FieldChip<F> {
    type Num = Number<F>;
    fn relu(
        &self,
        layouter: impl Layouter<F>,
        input: Self::Num,
        a: Self::Num,
        b: usize,
    ) -> Result<Self::Num, Error> {
        let config = self.config().relu_config.clone();

        let relu_chip = ReluChip::<F>::construct(config, ());
        relu_chip.relu(layouter, input, a, b)
    }
}

fn as_any_gen<T: 'static>(x: &T) -> &dyn Any {
    x
}

impl<F: Field> ReluInstructions<F> for ReluChip<F> {
    type Num = Number<F>;

    fn relu(
        &self,
        mut layouter: impl Layouter<F>,
        input: Self::Num,
        a: Self::Num,
        b: usize,
    ) -> Result<Self::Num, Error> {
        let config = self.config();

        layouter.assign_region(
            || "relu",
            |mut region: Region<'_, F>| {
                // We only want to use a single reluition gate in this region,
                // so we enable it at region offset 0; this means it will constrain
                // cells at offsets 0 and 1.
                config.s_relu.enable(&mut region, 0)?;

                input.0.copy_advice(|| "lhs", &mut region, config.advice[0], 0)?;
                a.0.copy_advice(|| "rhs", &mut region, config.advice[1], 0)?;

                let value = input.0.value().copied();
                let ca = value * a.0.value().copied();

                
                let impl_clip = |x: F| -> Value<F>
                {
                    let fp_val_x: &Fp =as_any_gen(&x).downcast_ref::<Fp>()
                    .expect("Error!");
                    let mut ind = 0;
                    if fp_val_x > &fp_val_x.neg(){
                        return Value::known(x-x);
                    } 
                    for i in 0..256{
                        if &Fp::from((b * i) as u64) > fp_val_x{
                            break;
                        }
                        ind +=  1;
                    }
                    ind -=  1;
                    Value::known(*as_any_gen(&Fp::from(ind)).downcast_ref::<F>().expect("Error!"))
                };

                let clipped = ca.and_then(impl_clip);

                region
                    .assign_advice(|| "lhs + rhs", config.advice[0], 1, || clipped)
                    .map(Number)
            },
        )
    }
}
// ANCHOR END: relu-instructions-impl

// ANCHOR: sub-instructions
trait SubInstructions<F: Field>: Chip<F> {
    /// Variable representing a number.
    type Num;

    /// Returns `c = a - b`.
    fn sub(
        &self,
        layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error>;
}
// ANCHOR_END: sub-instructions


// ANCHOR: sub-config
#[derive(Clone, Debug)]
struct SubConfig {
    advice: [Column<Advice>; 2],
    s_sub: Selector,
}
// ANCHOR_END: sub-config

// ANCHOR: sub-chip
struct SubChip<F: Field> {
    config: SubConfig,
    _marker: PhantomData<F>,
}
// ANCHOR END: sub-chip

// ANCHOR: sub-chip-trait-impl
impl<F: Field> Chip<F> for SubChip<F> {
    type Config = SubConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}
// ANCHOR END: sub-chip-trait-impl

// ANCHOR: sub-chip-impl
impl<F: Field> SubChip<F> {
    fn construct(config: <Self as Chip<F>>::Config, _loaded: <Self as Chip<F>>::Loaded) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 2],
    ) -> <Self as Chip<F>>::Config {
        let s_sub = meta.selector();

        // Define our subtraction gate!
        meta.create_gate("sub", |meta| {
            let lhs = meta.query_advice(advice[0], Rotation::cur());
            let rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[0], Rotation::next());
            let s_sub = meta.query_selector(s_sub);

            vec![s_sub * (lhs - rhs - out)]
        });

        SubConfig { advice, s_sub }
    }
}
// ANCHOR END: sub-chip-impl

// ANCHOR: sub-instructions-impl
impl<F: Field> SubInstructions<F> for FieldChip<F> {
    type Num = Number<F>;
    fn sub(
        &self,
        layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error> {
        let config = self.config().sub_config.clone();

        let sub_chip = SubChip::<F>::construct(config, ());
        sub_chip.sub(layouter, a, b)
    }
}

impl<F: Field> SubInstructions<F> for SubChip<F> {
    type Num = Number<F>;

    fn sub(
        &self,
        mut layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error> {
        let config = self.config();

        layouter.assign_region(
            || "sub",
            |mut region: Region<'_, F>| {
                config.s_sub.enable(&mut region, 0)?;

                a.0.copy_advice(|| "lhs", &mut region, config.advice[0], 0)?;
                b.0.copy_advice(|| "rhs", &mut region, config.advice[1], 0)?;

                let value = a.0.value().copied() - b.0.value();

                region
                    .assign_advice(|| "lhs - rhs", config.advice[0], 1, || value)
                    .map(Number)
            },
        )
    }
}
// ANCHOR END: sub-instructions-impl

// ANCHOR: mul-instructions
trait MulInstructions<F: Field>: Chip<F> {
    /// Variable representing a number.
    type Num;

    /// Returns `c = a * b`.
    fn mul(
        &self,
        layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error>;
}
// ANCHOR_END: mul-instructions

// ANCHOR: mul-config
#[derive(Clone, Debug)]
struct MulConfig {
    advice: [Column<Advice>; 2],
    s_mul: Selector,
}
// ANCHOR END: mul-config


// ANCHOR: mul-chip
struct MulChip<F: Field> {
    config: MulConfig,
    _marker: PhantomData<F>,
}
// ANCHOR_END: mul-chip

// ANCHOR: mul-chip-trait-impl
impl<F: Field> Chip<F> for MulChip<F> {
    type Config = MulConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}
// ANCHOR END: mul-chip-trait-impl

// ANCHOR: mul-chip-impl
impl<F: Field> MulChip<F> {
    fn construct(config: <Self as Chip<F>>::Config, _loaded: <Self as Chip<F>>::Loaded) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 2],
    ) -> <Self as Chip<F>>::Config {
        let s_mul = meta.selector();

        // Define our multiplication gate!
        meta.create_gate("mul", |meta| {
            // To implement multiplication, we need three advice cells and a selector
            // cell. We arrange them like so:
            //
            // | a0  | a1  | s_mul |
            // |-----|-----|-------|
            // | lhs | rhs | s_mul |
            // | out |     |       |
            //
            // Gates may refer to any relative offsets we want, but each distinct
            // offset adds a cost to the proof. The most common offsets are 0 (the
            // current row), 1 (the next row), and -1 (the previous row), for which
            // `Rotation` has specific constructors.
            let lhs = meta.query_advice(advice[0], Rotation::cur());
            let rhs = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[0], Rotation::next());
            let s_mul = meta.query_selector(s_mul);

            // The polynomial expression returned from `create_gate` will be
            // constrained by the proving system to equal zero. Our expression
            // has the following properties:
            // - When s_mul = 0, any value is allowed in lhs, rhs, and out.
            // - When s_mul != 0, this constrains lhs * rhs = out.
            vec![s_mul * (lhs * rhs - out)]
        });

        MulConfig { advice, s_mul }
    }
}
// ANCHOR_END: mul-chip-impl

// ANCHOR: mul-instructions-impl
impl<F: Field> MulInstructions<F> for FieldChip<F> {
    type Num = Number<F>;
    fn mul(
        &self,
        layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error> {
        let config = self.config().mul_config.clone();
        let mul_chip = MulChip::<F>::construct(config, ());
        mul_chip.mul(layouter, a, b)
    }
}

impl<F: Field> MulInstructions<F> for MulChip<F> {
    type Num = Number<F>;

    fn mul(
        &self,
        mut layouter: impl Layouter<F>,
        a: Self::Num,
        b: Self::Num,
    ) -> Result<Self::Num, Error> {
        let config = self.config();

        layouter.assign_region(
            || "mul",
            |mut region: Region<'_, F>| {
                // We only want to use a single multiplication gate in this region,
                // so we enable it at region offset 0; this means it will constrain
                // cells at offsets 0 and 1.
                config.s_mul.enable(&mut region, 0)?;

                // The inputs we've been given could be located anywhere in the circuit,
                // but we can only rely on relative offsets inside this region. So we
                // assign new cells inside the region and constrain them to have the
                // same values as the inputs.
                a.0.copy_advice(|| "lhs", &mut region, config.advice[0], 0)?;
                b.0.copy_advice(|| "rhs", &mut region, config.advice[1], 0)?;

                // Now we can compute the multiplication result, which is to be assigned
                // into the output position.
                let value = a.0.value().copied() * b.0.value();

                // Finally, we do the assignment to the output, returning a
                // variable to be used in another part of the circuit.
                region
                    .assign_advice(|| "lhs * rhs", config.advice[0], 1, || value)
                    .map(Number)
            },
        )
    }
}
// ANCHOR END: mul-instructions-impl


// ANCHOR: field-instructions
trait FieldInstructions<F: Field>: ReluInstructions<F> + AddInstructions<F> + MulInstructions<F> + SubInstructions<F> {
    /// Variable representing a number.
    type Num;

    /// Loads a number into the circuit as a private input.
    fn load_private(
        &self,
        layouter: impl Layouter<F>,
        a: Value<F>,
    ) -> Result<<Self as FieldInstructions<F>>::Num, Error>;
    
    // run the full CNN model, starting from CNN layers and ending with FC layers
    fn run_model(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>,
        kernels: Vec<Vec<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>>>,
        biases: Vec<Vec<<Self as FieldInstructions<F>>::Num>>,
        w_last_fc_layer: Vec<Vec<<Self as FieldInstructions<F>>::Num>>,
        bias_last_fc_layer: Vec<<Self as FieldInstructions<F>>::Num>,
        zs: Vec<<Self as FieldInstructions<F>>::Num>,
        scale_as: Vec<<Self as FieldInstructions<F>>::Num>,
        scale_b: usize,
        stride: usize,
    ) -> Result<Vec<<Self as FieldInstructions<F>>::Num>, Error>;

    // run relu for a vector inputs
    fn relu_vector(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>,
        a: <Self as FieldInstructions<F>>::Num,
        b: usize,
    ) -> Result<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>, Error>;

    // run conv2d like torch.nn.func.conv2d
    fn conv_2d(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>,
        kernel: Vec<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>>,
        bias: Vec<<Self as FieldInstructions<F>>::Num>,
        z: <Self as FieldInstructions<F>>::Num,
        stride: usize,
    ) -> Result<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>, Error>;

    // run dot product as the paper equation out = sum (i:0->N)[ (ai- z) * b] + bias
    fn dot_product(
        &self,
        layouter: &mut impl Layouter<F>,
        a: Vec<<Self as FieldInstructions<F>>::Num>,
        b: Vec<<Self as FieldInstructions<F>>::Num>,
        bias: <Self as FieldInstructions<F>>::Num,
        z: <Self as FieldInstructions<F>>::Num, 
    ) -> Result<<Self as FieldInstructions<F>>::Num, Error>;

    /// Exposes a number as a public input to the circuit.
    fn expose_public(
        &self,
        layouter: impl Layouter<F>,
        num: <Self as FieldInstructions<F>>::Num,
        row: usize,
    ) -> Result<(), Error>;
}
// ANCHOR_END: field-instructions

#[derive(Clone, Debug)]
struct FieldConfig {
    advice: [Column<Advice>; 2],

    /// Public inputs
    instance: Column<Instance>,
    add_config: AddConfig,
    sub_config: SubConfig,
    mul_config: MulConfig,
    relu_config: ReluConfig,
}

// ANCHOR: field-chip
/// The top-level chip that will implement the `FieldInstructions`.
struct FieldChip<F: Field> {
    config: FieldConfig,
    _marker: PhantomData<F>,
}
// ANCHOR_END: field-chip

// ANCHOR: field-chip-trait-impl
impl<F: Field> Chip<F> for FieldChip<F> {
    type Config = FieldConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}
// ANCHOR_END: field-chip-trait-impl

// ANCHOR: field-chip-impl
impl<F: Field> FieldChip<F> {
    fn construct(config: <Self as Chip<F>>::Config, _loaded: <Self as Chip<F>>::Loaded) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 2],
        instance: Column<Instance>,
    ) -> <Self as Chip<F>>::Config {
        let add_config = AddChip::configure(meta, advice);
        let sub_config = SubChip::configure(meta, advice);
        let mul_config = MulChip::configure(meta, advice);
        let relu_config = ReluChip::configure(meta, advice);
        for column in &advice {
            meta.enable_equality(*column);
        }

        meta.enable_equality(instance);

        FieldConfig {
            advice,
            instance,
            add_config,
            sub_config,
            mul_config,
            relu_config,
        }
    }
}
// ANCHOR_END: field-chip-impl

// ANCHOR: field-instructions-impl
impl<F: Field> FieldInstructions<F> for FieldChip<F> {
    type Num = Number<F>;

    fn load_private(
        &self,
        mut layouter: impl Layouter<F>,
        value: Value<F>,
    ) -> Result<<Self as FieldInstructions<F>>::Num, Error> {
        let config = self.config();

        layouter.assign_region(
            || "load private",
            |mut region| {
                region
                    .assign_advice(|| "private input", config.advice[0], 0, || value)
                    .map(Number)
            },
        )
    }
    
    fn run_model(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>,
        kernels: Vec<Vec<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>>>,
        biases: Vec<Vec<<Self as FieldInstructions<F>>::Num>>,
        w_last_fc_layer: Vec<Vec<<Self as FieldInstructions<F>>::Num>>,
        bias_last_fc_layer: Vec<<Self as FieldInstructions<F>>::Num>,
        zs: Vec<<Self as FieldInstructions<F>>::Num>,
        scale_as: Vec<<Self as FieldInstructions<F>>::Num>,
        scale_b: usize,
        stride: usize,
    ) -> Result<Vec<<Self as FieldInstructions<F>>::Num>, Error> {

        let mut conv_input = input;
        let mut zind = 0;
        // the CNN part
        for i in 0..kernels.len(){
            let conv_out = self.conv_2d(&mut layouter.namespace(|| "conv2d"), conv_input, kernels[i].clone(), biases[i].clone(), zs[zind].clone(), stride)?;
            conv_input = self.relu_vector( &mut layouter.namespace(|| "relu"), conv_out, scale_as[zind].clone(), scale_b)?;
            zind += 1;
        }

        let mut flatten = Vec::new();
        for i in 0..conv_input.len(){
            for j in 0..conv_input[0].len(){
                for k in 0..conv_input[0][0].len(){
                    flatten.push(conv_input[i][j][k].clone());
                }
            }
        }

        // the FC part
        let mut out = vec![self.dot_product(&mut layouter.namespace(|| "dot product"), flatten.clone(), w_last_fc_layer[0].clone(), bias_last_fc_layer[0].clone(), zs[zind].clone())?];
        for i in 1..bias_last_fc_layer.len(){
            out.push(self.dot_product(&mut layouter.namespace(|| "dot product"), flatten.clone(), w_last_fc_layer[i].clone(), bias_last_fc_layer[i].clone(), zs[zind].clone())?);
        }

        Ok(out)

    }

    fn relu_vector(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>,
        a: <Self as FieldInstructions<F>>::Num,
        b: usize,
    ) -> Result<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>, Error> {
        let value =self.relu(layouter.namespace(|| "relu(a)"), input[0][0][0].clone(), a.clone(), b)?;
        let mut out = vec![vec![vec![value.clone();input[0][0].len()];input[0].len()];input.len()];
        
        for k in 0..input.len() {
            for j in 0..input[k].len() {
                for i in 0..input[k][j].len() {
                    out[k][j][i] = self.relu(layouter.namespace(|| "relu(input)"), input[k][j][i].clone(), a.clone(), b)?;
                }
            }
        }
        Ok(out)
    }

    fn conv_2d(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>,
        kernel: Vec<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>>,
        bias: Vec<<Self as FieldInstructions<F>>::Num>,
        z: <Self as FieldInstructions<F>>::Num,
        stride: usize,
    ) -> Result<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>, Error> {
        let out_channels_size = kernel.len();
        let in_channels_size = kernel[0].len();
        let kernel_size = kernel[0][0].len();
        let out_size = (( input[0].len() - kernel_size ) / stride) + 1;

        let mut kernel_as_vec = Vec::new();
        for filter in kernel{
            let mut filter_as_vec = Vec::new();
            for i in 0..in_channels_size{
                for j in 0..kernel_size{
                    for k in 0..kernel_size{
                        filter_as_vec.push(filter[i][j][k].clone());
                    }
                }
            }
            kernel_as_vec.push(filter_as_vec);
        }

        let mut out = vec![vec![vec![bias[0].clone(); out_size]; out_size]; out_channels_size]; 
        
        let mut left_x: usize = 0;
        for out_x in 0..out_size{
            let mut top_y: usize = 0;
            for out_y in 0..out_size{


                let mut input_as_vec = Vec::new();
                for i in 0..in_channels_size{
                    for j in left_x..left_x+kernel_size{
                        for k in top_y..top_y+kernel_size{
                            input_as_vec.push(input[i][j][k].clone());
                        }
                    }
                }
                
                for i_filter in 0..kernel_as_vec.len(){
                    let filter_as_vec = kernel_as_vec[i_filter].clone();
                    let dot = self.dot_product(&mut layouter.namespace(|| "x * y"), input_as_vec.clone(), filter_as_vec, bias[i_filter].clone(), z.clone())?;
                    out[i_filter][out_x][out_y] = dot;

                }

                top_y = top_y + stride;
            }
            left_x = left_x + stride;
        }
        
        
        Ok(out)
    }

    fn dot_product(
        &self,
        layouter: &mut impl Layouter<F>,
        a: Vec<<Self as FieldInstructions<F>>::Num>,
        b: Vec<<Self as FieldInstructions<F>>::Num>,
        bias: <Self as FieldInstructions<F>>::Num,
        z: <Self as FieldInstructions<F>>::Num, 
    ) -> Result<<Self as FieldInstructions<F>>::Num, Error> {        
        let mut x = a[0].clone();
        let mut y = b[0].clone();
        
        let mut subx = self.sub(layouter.namespace(|| "x * y"), x, z.clone())?;
        let mut xy = self.mul(layouter.namespace(|| "x * y"), subx, y)?;
        let mut total = self.add(layouter.namespace(|| "a + b"), xy, bias)?;
        
        for i in 1..a.len() {
            x = a[i].clone();
            y = b[i].clone();
            subx = self.sub(layouter.namespace(|| "x * y"), x, z.clone())?;
            xy = self.mul(layouter.namespace(|| "x * y"), subx, y)?;
            total = self.add(layouter.namespace(|| "a + b"), xy, total)?;
        }
        
        Ok(total)
    }

    fn expose_public(
        &self,
        mut layouter: impl Layouter<F>,
        num: <Self as FieldInstructions<F>>::Num,
        row: usize,
    ) -> Result<(), Error> {
        let config = self.config();

        layouter.constrain_instance(num.0.cell(), config.instance, row)
    }
}
// ANCHOR_END: field-instructions-impl

// ANCHOR: circuit
/// The full circuit implementation.
///
/// In this struct we store the private input variables. We use `Value<F>` because
/// they won't have any value during key generation. During proving, if any of these
/// were `Value::unknown()` we would get an error.
#[derive(Default)]
struct MyCircuit<F: Field> {
    input: Vec<Vec<Vec<Value<F>>>>,
    kernels: Vec<Vec<Vec<Vec<Vec<Value<F>>>>>>,
    biases: Vec<Vec<Value<F>>>,
    w_last_fc_layer: Vec<Vec<Value<F>>>,
    bias_last_fc_layer: Vec<Value<F>>,
    scale_zs: Vec<Value<F>>,
    scale_as: Vec<Value<F>>,
    scale_b: usize, 
    stride: usize,
}

impl<F: Field> Circuit<F> for MyCircuit<F> {
    // Since we are using a single chip for everything, we can just reuse its config.
    type Config = FieldConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // We create the two advice columns that FieldChip uses for I/O.
        let advice = [meta.advice_column(), meta.advice_column()];

        // We also need an instance column to store public inputs.
        let instance = meta.instance_column();

        FieldChip::configure(meta, advice, instance)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let _new_dummy = 2;
        let field_chip = FieldChip::<F>::construct(config, ());
        // Load our private values into the circuit.

        // input: Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>,
        let loaded_v = field_chip.load_private(layouter.namespace(|| "load a"), self.input[0][0][0])?;
        let mut input = vec![vec![vec![loaded_v.clone(); self.input[0][0].len()]; self.input[0][0].len()]; self.input.len()]; 
        for i in 0..self.input.len() {
            for j in 0..self.input[i].len() {
                for k in 0..self.input[i][j].len() {
                    input[i][j][k] = field_chip.load_private(layouter.namespace(|| "load a"), self.input[i][j][k])?;
                }
            }
        }
        
        // kernels: Vec<Vec<Vec<Vec<Vec<<Self as FieldInstructions<F>>::Num>>>>>,
        let mut kernels = Vec::new() ;
        for i in 0..self.kernels.len() {
            let mut one_kernel = vec![vec![vec![vec![loaded_v.clone(); self.kernels[i][0][0][0].len()]; self.kernels[i][0][0].len()]; self.kernels[i][0].len()]; self.kernels[i].len()]; 
            for j in 0..self.kernels[i].len() {
                for k in 0..self.kernels[i][j].len() {
                    for l in 0..self.kernels[i][j][k].len() {
                        for o in 0..self.kernels[i][j][k][l].len() {
                            one_kernel[j][k][l][o] = field_chip.load_private(layouter.namespace(|| "load a"), self.kernels[i][j][k][l][o])?;
                        }
                    }
                }
            }
            kernels.push(one_kernel);
        }
        
        // biases: Vec<Vec<<Self as FieldInstructions<F>>::Num>>,
        let mut biases = Vec::new(); 
        for i in 0..self.biases.len(){
            let mut one_bias = vec![loaded_v.clone(); self.biases[i].len()];
            for j in 0..self.biases[i].len(){
                one_bias[j] = field_chip.load_private(layouter.namespace(|| "load bias"), self.biases[i][j])?;
            }
            biases.push(one_bias);
        }
        
        // w_last_fc_layer: Vec<Vec<<Self as FieldInstructions<F>>::Num>>,
        let mut w_last_fc_layer = vec![vec![loaded_v.clone(); self.w_last_fc_layer[0].len()]; self.w_last_fc_layer.len()];
        for i in 0..self.w_last_fc_layer.len(){
            for j in 0..self.w_last_fc_layer[i].len(){
                w_last_fc_layer[i][j] = field_chip.load_private(layouter.namespace(|| "load bias"), self.w_last_fc_layer[i][j])?;
            }
        }
        

        // bias_last_fc_layer: Vec<<Self as FieldInstructions<F>>::Num>,
        let mut bias_last_fc_layer = vec![loaded_v.clone(); self.bias_last_fc_layer.len()];
        for i in 0..self.bias_last_fc_layer.len(){
            bias_last_fc_layer[i] = field_chip.load_private(layouter.namespace(|| "load bias"), self.bias_last_fc_layer[i])?;
        }

        // scale_zs: Vec<<Self as FieldInstructions<F>>::Num>,
        let mut scale_zs = vec![loaded_v.clone(); self.scale_zs.len()];
        for i in 0..self.scale_zs.len(){
            scale_zs[i] = field_chip.load_private(layouter.namespace(|| "load bias"), self.scale_zs[i])?;
        }

        // scale_as: Vec<<Self as FieldInstructions<F>>::Num>,
        let mut scale_as = vec![loaded_v.clone(); self.scale_as.len()];
        for i in 0..self.scale_as.len(){
            scale_as[i] = field_chip.load_private(layouter.namespace(|| "load bias"), self.scale_as[i])?;
        }

        
        let e = field_chip.run_model(&mut layouter, input, kernels, biases, w_last_fc_layer, bias_last_fc_layer.clone(), scale_zs, scale_as, self.scale_b, self.stride)?;
        
        // print the last layer outputs
        for i in 0..e.len(){
            println!("i = {:?} e {:?}",i, e[i].0.value());
        }

        // Expose the result as a public input to the circuit.
        field_chip.expose_public(layouter.namespace(|| "expose e"), e[0].clone(), 0)
    }
}
// ANCHOR_END: circuit


fn load_from_file(file_path: &str) -> Vec<usize> {
    use std::fs::File;
    use std::io::BufReader;
    use std::io::BufRead;
    let file = File::open(file_path).expect("file wasn't found.");
    let reader = BufReader::new(file);

    let numbers: Vec<usize> = reader
        .lines()
        .map(|line| line.unwrap().parse::<usize>().unwrap())
        .collect();
    numbers
}

#[allow(clippy::many_single_char_names)]
fn main() {
    use halo2_proofs::{dev::MockProver};
    use std::env;
    env::set_var("RUST_BACKTRACE", "1");

    // ANCHOR: test-circuit
    let k = 17;


    // load all parameters from txt files
    let loaded_input_int = load_from_file("/Users/abdokaseb/Desktop/halo2task/halo2/halo2_proofs/examples/inputs.txt");

    let loaded_kernels = [load_from_file("/Users/abdokaseb/Desktop/halo2task/halo2/halo2_proofs/examples/f0.txt"), load_from_file("/Users/abdokaseb/Desktop/halo2task/halo2/halo2_proofs/examples/f1.txt"), load_from_file("/Users/abdokaseb/Desktop/halo2task/halo2/halo2_proofs/examples/f2.txt")];
    
    let loaded_w_last_fc_layer_int = load_from_file("/Users/abdokaseb/Desktop/halo2task/halo2/halo2_proofs/examples/w1.txt");    
    
    let other_params = load_from_file("/Users/abdokaseb/Desktop/halo2task/halo2/halo2_proofs/examples/other_parm.txt");

    let stride: usize = other_params[0];
    
    let mut biases = Vec::new();
    let mut ind = 1;
    for _i in 0..other_params[1]{
        ind += 1;
        let mut bias = vec![Value::known(Fp::from(0)); other_params[ind]];
        for j in 0..other_params[ind]{
            ind += 1;
            bias[j] = Value::known(Fp::from(other_params[ind] as u64));
        }
        biases.push(bias);
    }

    ind += 1;
    let mut bias_last_fc_layer = vec![Value::known(Fp::from(0)); other_params[ind]];
    for i in 0..other_params[ind]{
        ind += 1;
        bias_last_fc_layer[i] = Value::known(Fp::from(other_params[ind] as u64));
    }    
    
    ind += 1;
    let mut scale_z = vec![Value::known(Fp::from(0)); other_params[ind]];
    for i in 0..other_params[ind]{
        ind += 1;
        scale_z[i] = Value::known(Fp::from(other_params[ind] as u64));
    }    
    
    ind += 1;
    let mut scale_a = vec![Value::known(Fp::from(0)); other_params[ind]];
    for i in 0..other_params[ind]{
        ind += 1;
        scale_a[i] = Value::known(Fp::from(other_params[ind] as u64));
    }    
    
    ind += 1;
    let scale_b: usize = other_params[ind];
    
    ind += 1;
    let e_int = other_params[ind] as u64;


    let mut input = vec![vec![vec![Value::known(Fp::from(0)); loaded_input_int[3]]; loaded_input_int[2]]; loaded_input_int[1]]; 
    let mut w_last_fc_layer = vec![vec![Value::known(Fp::from(0));loaded_w_last_fc_layer_int[2]];loaded_w_last_fc_layer_int[1]];
    
    let mut ind = 4;
    for i in 0..loaded_input_int[1]{
        for j in 0..loaded_input_int[2]{
            for k in 0..loaded_input_int[3]{
                input[i][j][k] = Value::known(Fp::from(loaded_input_int[ind] as u64));
                ind += 1;
            }
        }
    }
    
    let mut ind = 3;
    for i in 0..loaded_w_last_fc_layer_int[1]{
        for j in 0..loaded_w_last_fc_layer_int[2]{
            w_last_fc_layer[i][j] = Value::known(Fp::from(loaded_w_last_fc_layer_int[ind] as u64));
            ind += 1;
        }
    }
    
    let mut kernels = Vec::new();
    
    for i_kernels in 0..loaded_kernels.len(){
        let mut kernel = vec![vec![vec![vec![Value::known(Fp::from(0)); loaded_kernels[i_kernels][4]]; loaded_kernels[i_kernels][3]]; loaded_kernels[i_kernels][2]]; loaded_kernels[i_kernels][1]]; 
        
        let mut ind = 5;
        for i in 0..loaded_kernels[i_kernels][1]{
            for j in 0..loaded_kernels[i_kernels][2]{
                for k in 0..loaded_kernels[i_kernels][3]{
                    for l in 0..loaded_kernels[i_kernels][4]{
                        kernel[i][j][k][l] = Value::known(Fp::from(loaded_kernels[i_kernels][ind] as u64));
                        ind += 1;
                    }
                }
            }
        }
        kernels.push(kernel);
    }

    let e = Fp::from(e_int);

    // Instantiate the circuit with the private input.
    let circuit = MyCircuit {
        input: input,
        kernels: kernels,
        biases: biases,
        w_last_fc_layer: w_last_fc_layer,
        bias_last_fc_layer: bias_last_fc_layer,
        scale_zs: scale_z,
        scale_as: scale_a,
        scale_b: scale_b,
        stride: stride,
    };

    // Arrange the public input. We expose the result in row 0
    // of the instance column, so we position it there in our public inputs.
    let public_inputs = vec![e];

    // Given the correct public input, our circuit will verify.
    let prover = MockProver::run(k, &circuit, vec![public_inputs.clone()]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
    // ANCHOR_END: test-circuit

    println!("End Model Run");
}