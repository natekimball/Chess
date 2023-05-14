use std::{sync::Arc, path::Path, error::Error};
// use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Session, Tensor, Operation, SavedModelBuilder, SavedModelSaver, Scope, ImportGraphDefOptions, MetaGraphDef, SignatureDef, REGRESS_INPUTS, REGRESS_METHOD_NAME, REGRESS_OUTPUTS, Output, Status, Variable, ops, DataType};

// const REGRESS_METHOD_NAME: &str = "serve";

#[derive(Clone)]
pub struct Model {
    // session: Arc<Session>,
    // input_op_train: Operation,
    // target_op_train: Operation,
    // output_op_train: Operation,
    // input_op_pred: Operation,
    // output_op_pred: Operation,
    // save_dir: String,
    // signature_train: SignatureDef,
    // signature_pred: SignatureDef
}

impl Model {
    pub fn new() -> Self {
        // let input_parameter_name = "input";
        // let output_parameter_name = "output_0";
        // let train_input_target_name = "training_target";
        // let save_dir = "model/saved_model";
        // let mut graph = Graph::new();
        // let bundle = SavedModelBundle::load(
        //     &SessionOptions::new(), &["serve"], &mut graph, Model::get_current_version(save_dir)
        // ).expect("Can't load saved model");
        // // println!("sigs: {:?}", bundle.meta_graph_def().signatures());
        
        // let signature_train = bundle.meta_graph_def().get_signature("train").expect("signature error").clone();
        // let input_info_train = signature_train.get_input(input_parameter_name).expect("signature error");
        // let target_info_train = signature_train.get_input(train_input_target_name).expect("signature error");
        // let output_info_train = signature_train.get_output(output_parameter_name).expect("signature error");  
        // let input_op_train = graph.operation_by_name_required(&input_info_train.name().name).expect("operation error");
        // let target_op_train = graph.operation_by_name_required(&target_info_train.name().name).expect("operation error");
        // let output_op_train = graph.operation_by_name_required(&output_info_train.name().name).expect("operation error");
        

        // let signature_pred = bundle.meta_graph_def().get_signature("pred").expect("signature error").clone();
        // let input_info_pred = signature_train.get_input(input_parameter_name).expect("signature error");
        // let output_info_pred = signature_train.get_output(output_parameter_name).expect("signature error");
        // let input_op_pred = graph.operation_by_name_required(&input_info_pred.name().name).expect("operation error");
        // let output_op_pred = graph.operation_by_name_required(&output_info_pred.name().name).expect("operation error");


        
        Self {
            // session: Arc::new(bundle.session),
            // input_op_train,
            // target_op_train,
            // output_op_train,
            // input_op_pred,
            // output_op_pred,
            // save_dir: save_dir.to_string(),
            // signature_train,
            // signature_pred
        }
    }

    pub fn run_inference(&self, input_data: &Vec<[[[f32; 8]; 8]; 13]>) -> Result<Vec<f32>, String> {
        // let len = input_data.len() as u64;
        // let now = std::time::SystemTime::now();

        // let data = input_data.clone().into_iter().flatten().flatten().flatten().collect::<Vec<f32>>();
        // let input_tensor: Tensor<f32> = Tensor::new(&[len,13,8,8]).with_values(&data)?;

        // let mut args = SessionRunArgs::new();
        // args.add_feed(&self.input_op_pred, 0, &input_tensor);
    
        // let out = args.request_fetch(&self.output_op_pred, 0);
    
        // self.session
        // .run(&mut args)
        // .expect("Error occurred during inference");
    
        // let prediction = args.fetch(out)?;
        // // println!("data : {:?}", input_tensor);
        // let elapsed = now.elapsed().unwrap();
        // println!("time to inference {len} inputs: {:?}, {}ms/inf", elapsed, elapsed.as_millis() / len as u128);
        // // println!("Prediction: {:?}", prediction);
        
        // Ok(prediction.to_vec())
        Err("this method shouldn't have been called".to_string())
    }

    pub fn back_propagate(&self, input_data: &Vec<[[[f32; 8]; 8]; 13]>, amplified_scores: &Vec<f32>) {
        // let len = input_data.len() as u64;
        // let now = std::time::SystemTime::now();

        // let data = input_data.clone().into_iter().flatten().flatten().flatten().collect::<Vec<f32>>();
        // let input_tensor: Tensor<f32> = Tensor::new(&[len,13,8,8]).with_values(&data).expect("Can't create tensor from input data");
        // let target_tensor: Tensor<f32> = Tensor::new(&[len,1]).with_values(&amplified_scores).expect("Can't create tensor from target data");

        // let mut args = SessionRunArgs::new();

        // args.add_feed(&self.input_op_train, 0, &input_tensor);
        // args.add_feed(&self.target_op_train, 0, &target_tensor);

        // let out = args.request_fetch(&self.output_op_train, 0);

        // self.session
        // .run(&mut args)
        // .expect("Error occurred during calculations");
        
        // let loss: f32 = args.fetch(out).unwrap()[0];

        // let elapsed = now.elapsed().unwrap();
        // println!("time to train on {len} inputs: {:?}, {}ms/input", elapsed, elapsed.as_millis() / len as u128);
        // println!("Loss: {:?}", loss);
    }

    // pub fn save_model(&self) {
    //     let save_path = Path::new(&self.save_dir);
    //     let export_dir = save_path.join("_v");
    //     let version = Model::get_next_version(&export_dir);

    //     if let Err(e) = std::fs::create_dir_all(&export_dir) {
    //         eprintln!("Failed to create export directory: {:?}", e);
    //         return;
    //     }

    //     let mut builder = SavedModelBuilder::new();
    //     builder.add_signature("pred", self.signature_train.clone());
    //     builder.add_signature("train", self.signature_pred.clone());
        
    //     let control_deps = [self.input_op_train.clone(), self.target_op_train.clone(), self.output_op_train.clone(), self.input_op_pred.clone(), self.output_op_pred.clone()];
        
    //     // let mut scope: Scope = Scope::new_root_scope().with_control_dependencies(&control_deps);
    //     let mut scope = Scope::new_root_scope();
        
    //     let saver = builder.inject(&mut scope).expect("failed to create SavedModelSaver");
        
    //     // // let graph_def = self.graph.graph_def().unwrap();
    //     // // let graph_def = graph_def.as_slice();
    //     // // let options = ImportGraphDefOptions::new();
    //     // // let mut graph = Graph::new();
    //     // // graph.import_graph_def(graph_def, &options).unwrap();       

    //     // saver.save(&self.session, &self.graph, version).expect("failed to save model");


    // }

    // fn get_next_version(export_dir: &Path) -> String {
    //     let mut version = 1;
    //     while export_dir.join(version.to_string()).exists() {
    //         version += 1;
    //     }
    //     version.to_string()
    // }

    // fn get_current_version(save_dir: &str) -> String {
    //     let save_dir = Path::new(save_dir);
    //     if save_dir.exists() {
    //         return save_dir.to_str().unwrap().to_string();
    //     }
    //     let mut version = 0;
    //     let save_dir = save_dir.join("_v");
    //     while save_dir.join(version.to_string()).exists() {
    //         version += 1;
    //     }
    //     version.to_string()
    // }

//     pub fn build(mut self) -> Result<(), Box<dyn Error>> {
//         let (vars0, layer0) = layer(
//             self.input_op_train.clone(),
//             self.input_op_train.num_inputs() as u64,
//             *self.hidden.first().unwrap_or(&self.output),
//             &Self::relu,
//             &mut self.scope,
//         )?;
//         let (vars_hidden, layer_hidden) = {
//             let mut vars = Vec::new();
//             let mut last_size = *self.hidden.first().unwrap_or(&self.output);
//             let mut last_layer = layer0;
//             if let Some(hidden) = self.hidden.get(1..) {
//                 for &hsize in hidden {
//                     let (vars_n, layer_n) = layer(
//                         last_layer,
//                         last_size,
//                         hsize,
//                         &Self::relu,
//                         &mut self.scope,
//                     )?;
//                     vars.extend(vars_n);
//                     last_size = hsize;
//                     last_layer = layer_n;
//                 }
//             }
//             (vars, last_layer)
//         };
//         let (vars_output, layer_output) = layer(
//             layer_hidden,
//             *self.hidden.last().unwrap(),
//             self.output,
//             &Self::pass,
//             &mut self.scope,
//         )?;
//         let error = ops::sub(layer_output.clone(), output.clone(), &mut self.scope)?;
//         let error_squared = ops::mul(error.clone(), error, &mut self.scope)?;
//         let variables = vars0.into_iter()
//             .chain(vars_hidden)
//             .chain(vars_output)
//             .collect::<Vec<_>>();
//         let (minimizer_vars, minimize) = self.optimizer.minimize(
//             &mut self.scope,
//             error_squared.clone().into(),
//             MinimizeOptions::default().with_variables(&variables),
//         )?;
//         let mut all_vars = variables.clone();
//         all_vars.extend_from_slice(&minimizer_vars);
//         let mut builder = tensorflow::SavedModelBuilder::new();
//         builder
//             .add_collection("train", &all_vars)
//             .add_tag("serve")
//             .add_tag("train")
//             .add_signature(REGRESS_METHOD_NAME, {
//                 let mut def = SignatureDef::new(REGRESS_METHOD_NAME.to_string());
//                 def.add_input_info(
//                     REGRESS_INPUTS.to_string(),
//                     TensorInfo::new(
//                         DataType::Float,
//                         Shape::from(None),
//                         OutputName {
//                             name: input.name()?,
//                             index: 0,
//                         },
//                     ),
//                 );
//                 def.add_output_info(
//                     REGRESS_OUTPUTS.to_string(),
//                     TensorInfo::new(
//                         DataType::Float,
//                         Shape::from(None),
//                         layer_output.name()?,
//                     ),
//                 );
//                 def
//             });
//         let saver = builder.inject(&mut self.scope)?;
//         let options = SessionOptions::new();
//         // let session = Session::new(&options, &self.scope.graph())?;
//     }
// }

// fn layer<O: Into<Output>>(
//     input: O,
//     input_size: u64,
//     output_size: u64,
//     activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
//     scope: &mut Scope,
// ) -> Result<(Vec<Variable>, Output), Status>
// {
//     let scope = &mut scope.new_sub_scope("layer");
//     let w_shape = ops::constant(&[input_size as i64, output_size as i64], scope)?;
//     let w = Variable::builder()
//         .initial_value(
//             ops::RandomStandardNormal::new()
//                 .dtype(DataType::Float)
//                 .build(w_shape, scope)?,
//         )
//         .data_type(DataType::Float)
//         .shape([input_size, output_size])
//         .build(&mut scope.with_op_name("w"))?;
//     let b = Variable::builder()
//         .const_initial_value(Tensor::<f32>::new(&[output_size]))
//         .build(&mut scope.with_op_name("b"))?;
//     Ok((
//         vec![w.clone(), b.clone()],
//         activation(
//             ops::add(
//                 ops::mat_mul(input, w.output().clone(), scope)?,
//                 b.output().clone(),
//                 scope,
//             )?
//                 .into(),
//             scope,
//         )?,
//     ))
}

// let saved_model_saver = builder.inject(&mut self.scope)?;
//         self.SavedModelSaver.replace(Some(saved_model_saver));
//     }

//     if !Path::new(name).exists() {
//         fs::create_dir(name)?;
//     }
//     self.SavedModelSaver.borrow_mut().as_mut().unwrap().save(
//         &self.session,
//         &self.scope.graph(),
//         &format!("{}/{}", name, self.name),
//     )?;
//     self.serialize_network(format!("{}/{}", name, self.name))?;

//     Ok(())
// }


// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_load_and_save_model() {
//         let model = Model::new();
//         model.save_model();
//         assert!(true);
//     }
// }

        // let signature_save = bundle.meta_graph_def().get_signature("save").expect("signature error").clone();
        // let save_info = signature_save.get


//         let op_file_path = graph.operation_by_name_required("save/Const")?;
// let op_save = graph.operation_by_name_required("save/control_dependency")?;
// let file_path_tensor: Tensor<String> = Tensor::from(String::from(ckpt_file_path));
// let mut step = StepWithGraph::new();
// step.add_input(&op_file_path, 0, &file_path_tensor);
// step.add_target(&op_save);
// session.run(&mut step)?;


