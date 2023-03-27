// tract simply doesn't work with CNNs
// none of tensorflow, onnx, or pytorch can derive clone
// running into issues with pytorch and onnx
// onnx: try changing opset or change onnx_model.ir_version = 4
// maybe use tensorflow with arc?
// maybe use tensorflow but pass a (arc?) reference to the model from main?
// maybe use tensorflow or pytorch and define model in main (or other creational pattern) and pass it into game.rs

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

#[derive(Clone)]
pub struct Model;
// pub struct Model<'a> {
    // pred_input_parameter_name: String,
    // pred_output_parameter_name: String,
    // save_dir: String,
    // graph: &'a Graph,
    // bundle: &'a SavedModelBundle,
// }

impl Model {
// impl<'a> Model<'a> {
    // pub fn new(bundle: &'a SavedModelBundle, graph: &'a Graph) -> Self {
    //     let pred_input_parameter_name = "conv2d_input".to_owned();
    //     let pred_output_parameter_name = "dense_4".to_owned();
    //     let save_dir = "model/saved_model".to_owned();
    //     Self {
    //         pred_input_parameter_name,
    //         pred_output_parameter_name,
    //         save_dir,
    //         graph,
    //         bundle,
    //     }
    // }

    pub fn run_inference(input_data: [[[u8; 8]; 8]; 13]) -> Result<f32, Box<dyn std::error::Error>> {
        let pred_input_parameter_name = "conv2d_input";
        let pred_output_parameter_name = "dense_4";
    
        //Create some tensors to feed to the model for training, one as input and one as the target value
        //Note: All tensors must be declared before args!
        let input_tensor: Tensor<f32> = Tensor::new(&[1,13,8,8]).with_values(&input_data.into_iter().flatten().flatten().map(|u| u as f32).collect::<Vec<f32>>()).unwrap();
    
        //Path of the saved model
        let save_dir = "model/saved_model";
    
        //Create a graph
        let mut graph = Graph::new();
    
        //Load save model as graph
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, save_dir
        ).expect("Can't load saved model");
    
        //Initiate a session
        let session = &bundle.session;
    
        //Alternative to saved_model_cli. This will list all signatures in the console when run
        // let sigs = bundle.meta_graph_def().signatures();
        // println!("sigs: {:?}", sigs);

    
        //Retrieve the pred functions signature
        let signature_train = bundle.meta_graph_def().get_signature("serving_default").unwrap();
        let input_info_pred = signature_train.get_input(pred_input_parameter_name).unwrap();
        let output_info_pred = signature_train.get_output(pred_output_parameter_name).unwrap();
        let input_op_pred = graph.operation_by_name_required(&input_info_pred.name().name).unwrap();
        let output_op_pred = graph.operation_by_name_required(&output_info_pred.name().name).unwrap();
    
        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op_pred, 0, &input_tensor);
    
        let out = args.request_fetch(&output_op_pred, 0);
    
        //Run the session
        session
        .run(&mut args)
        .expect("Error occurred during inference");
    
        let prediction = args.fetch(out)?;
    
        println!("data : {:?}", input_tensor);
        println!("Prediction: {:?}", prediction);
        
        Ok(prediction[0])
    }
}


#[allow(dead_code)]
fn train() {
    //Sigmatures declared when we saved the model
    let train_input_parameter_input_name = "training_input";
    let train_input_parameter_target_name = "training_target";
    //Names of output nodes of the graph, retrieved with the saved_model_cli command
    let train_output_parameter_name = "output_0";

    //Create some tensors to feed to the model for training, one as input and one as the target value
    //Note: All tensors must be declared before args!
    let input_tensor: Tensor<f32> = Tensor::new(&[1,2]).with_values(&[1.0, 1.0]).unwrap();
    let target_tensor: Tensor<f32> = Tensor::new(&[1,1]).with_values(&[2.0]).unwrap();

    //Path of the saved model
    let save_dir = "model/tf_saved_model";

    //Create a graph
    let mut graph = Graph::new();

    //Load save model as graph
    let bundle = SavedModelBundle::load(
        &SessionOptions::new(), &["serve"], &mut graph, save_dir
    ).expect("Can't load saved model");

    //Initiate a session
    let session = &bundle.session;

    //Alternative to saved_model_cli. This will list all signatures in the console when run
    // let sigs = bundle.meta_graph_def().signatures();
    // println!("{:?}", sigs);
    

    //Retrieve the train functions signature
    let signature_train = bundle.meta_graph_def().get_signature("train").unwrap();

    //Input information
    let input_info_train = signature_train.get_input(train_input_parameter_input_name).unwrap();
    let target_info_train = signature_train.get_input(train_input_parameter_target_name).unwrap();

    //Output information
    let output_info_train = signature_train.get_output(train_output_parameter_name).unwrap();

    //Input operation
    let input_op_train = graph.operation_by_name_required(&input_info_train.name().name).unwrap();
    let target_op_train = graph.operation_by_name_required(&target_info_train.name().name).unwrap();

    //Output operation
    let output_op_train = graph.operation_by_name_required(&output_info_train.name().name).unwrap();

    //The values will be fed to and retrieved from the model with this
    let mut args = SessionRunArgs::new();

    //Feed the tensors into the graph
    args.add_feed(&input_op_train, 0, &input_tensor);
    args.add_feed(&target_op_train, 0, &target_tensor);

    //Fetch result from graph
    let out = args.request_fetch(&output_op_train, 0);

    //Run the session
    session
    .run(&mut args)
    .expect("Error occurred during calculations");

    //Retrieve the result of the operation
    let loss: f32 = args.fetch(out).unwrap()[0];

    println!("Loss: {:?}", loss);
}
    
// https://github.com/Grimmp/RustTensorFlowTraining

// fn run_regression(input_data: [[[u8; 8]; 8]; 13]) {

//     //Sigmatures declared when we saved the model
//     let train_input_parameter_input_name = "training_input";
//     let train_input_parameter_target_name = "training_target";
//     let pred_input_parameter_name = "inputs";

//     //Names of output nodes of the graph, retrieved with the saved_model_cli command
//     let train_output_parameter_name = "output_0";
//     let pred_output_parameter_name = "output_0";

//     //Create some tensors to feed to the model for training, one as input and one as the target value
//     //Note: All tensors must be declared before args!
//     let input_tensor: Tensor<f32> = Tensor::new(&[1,2]).with_values(&[1.0, 1.0]).unwrap();
//     let target_tensor: Tensor<f32> = Tensor::new(&[1,1]).with_values(&[2.0]).unwrap();

//     //Path of the saved model
//     let save_dir = "model/tf_saved_model";

//     //Create a graph
//     let mut graph = Graph::new();

//     //Load save model as graph
//     let bundle = SavedModelBundle::load(
//         &SessionOptions::new(), &["serve"], &mut graph, save_dir
//     ).expect("Can't load saved model");

//     //Initiate a session
//     let session = &bundle.session;

//     //Alternative to saved_model_cli. This will list all signatures in the console when run
//     // let sigs = bundle.meta_graph_def().signatures();
//     // println!("{:?}", sigs);
    

//     //Retrieve the train functions signature
//     let signature_train = bundle.meta_graph_def().get_signature("train").unwrap();

//     //Input information
//     let input_info_train = signature_train.get_input(train_input_parameter_input_name).unwrap();
//     let target_info_train = signature_train.get_input(train_input_parameter_target_name).unwrap();

//     //Output information
//     let output_info_train = signature_train.get_output(train_output_parameter_name).unwrap();

//     //Input operation
//     let input_op_train = graph.operation_by_name_required(&input_info_train.name().name).unwrap();
//     let target_op_train = graph.operation_by_name_required(&target_info_train.name().name).unwrap();

//     //Output operation
//     let output_op_train = graph.operation_by_name_required(&output_info_train.name().name).unwrap();

//     //The values will be fed to and retrieved from the model with this
//     let mut args = SessionRunArgs::new();

//     //Feed the tensors into the graph
//     args.add_feed(&input_op_train, 0, &input_tensor);
//     args.add_feed(&target_op_train, 0, &target_tensor);

//     //Fetch result from graph
//     let mut out = args.request_fetch(&output_op_train, 0);

//     //Run the session
//     session
//     .run(&mut args)
//     .expect("Error occurred during calculations");

//     //Retrieve the result of the operation
//     let loss: f32 = args.fetch(out).unwrap()[0];

//     println!("Loss: {:?}", loss);


//     //Retrieve the pred functions signature
//     let signature_train = bundle.meta_graph_def().get_signature("pred").unwrap();

//     //
//     let input_info_pred = signature_train.get_input(pred_input_parameter_name).unwrap();

//     //
//     let output_info_pred = signature_train.get_output(pred_output_parameter_name).unwrap();

//     //
//     let input_op_pred = graph.operation_by_name_required(&input_info_pred.name().name).unwrap();

//     //
//     let output_op_pred = graph.operation_by_name_required(&output_info_pred.name().name).unwrap();

//     args.add_feed(&input_op_pred, 0, &input_tensor);

//     out = args.request_fetch(&output_op_pred, 0);

//     //Run the session
//     session
//     .run(&mut args)
//     .expect("Error occurred during inference");

//     let prediction: f32 = args.fetch(out).unwrap()[0];

//     println!("Prediction: {prediction}");
    
//     prediction
// }



// use ndarray::prelude::*;
// use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor};

// // #[derive(Clone)]
// pub struct Model;

// impl Model {
//     pub fn run_inference(input_data: [[[u8; 8]; 8]; 13]) -> Result<f32, Box<dyn std::error::Error>> {

//         let model_path = "model/model.onnx";
//         let input_shape = [1, 13, 8, 8];

//         // Initialize the ONNX Runtime environment
//         let environment = Environment::builder()
//             .with_name("onnx_environment")
//             .build()
//             .unwrap();

//         // Load the ONNX model
//         let mut session = environment
//             .new_session_builder()
//             .unwrap()
//             .with_model_from_file(model_path)
//             .unwrap();
//         // Create a dummy input tensor with the specified shape
//         // let input_tensor = ndarray::Array::random(input_shape, ndarray_rand::RandomExt::standard_normal);
//         let input_tensor = Array::from(input_data.into_iter().flatten().flatten().collect::<Vec<u8>>()).into_shape(input_shape).unwrap();
        
//         // Create input name and value
//         // let input_name = self.session.input_names()[0].clone();
//         let input_value = input_tensor.into_dyn();
        
//         // Run inference
//         let outputs: Vec<OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> = session
//             .run(vec!(input_value))
//             .unwrap();

//         // Print the output tensor
//         for (i, output) in outputs.iter().enumerate() {
//             println!("Output {}: {:?}", i, output);
//         }
//         // outputs[0].as_slice().unwrap()[0].into()
//         Ok(outputs[0].as_slice().unwrap()[0])
//     }
// }



// // use ndarray::Array4;
// use tch::Tensor;
// use tch::CModule;

// // #[derive(Clone)]
// pub struct Model;

// impl Model {
//     pub fn run_inference(input_data: [[[u8; 8]; 8]; 13]) -> Result<f32, Box<dyn std::error::Error>> {
//         // Create a dummy input tensor with the shape [1, 13, 8, 8]
//         // let input_data: Vec<f32> = vec![0.0; 1 * 13 * 8 * 8];
//         let model = CModule::load("model/model.pt").unwrap();
//         println!("Model loaded");
//         let input_data = input_data.into_iter().flatten().flatten().collect::<Vec<u8>>();
//         let input_tensor = Tensor::of_slice(&input_data).reshape(&[1, 13, 8, 8]);
    
//         // Run the inference
//         let output = model.forward_ts(&[input_tensor]).unwrap();
    
//         // Convert the output tensor to ndarray
//         let output_data: Vec<f32> = output.into();
        
//         // let output_shape = [1;4];
//         // let output_array = Array4::from_shape_vec(output_shape, output_data).unwrap();
//         // println!("Inference result: {:?}", output_array);

//         println!("Inference result: {:?}", output_data);
//         Ok(output_data[0])
//     }
// }



// use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

// fn train() {
//     // Set device (use GPU if available)
//     let device = Device::cuda_if_available();

//     // Load your chess data and split into training and validation sets
//     // ...

//     // Create the CNN model
//     let vs = nn::VarStore::new(device);
//     let net = create_cnn(&vs.root());

//     // Set the optimizer and learning rate
//     let sgd = nn::Sgd::default().learning_rate(0.01).build(&vs, 0.01).unwrap();

//     // Train the model
//     for epoch in 1..=100 {
//         // Training loop
//         // ...

//         // Validation loop
//         // ...

//         println!("Epoch: {}", epoch);
//     }
// }

// fn create_cnn(p: &nn::Path) -> impl nn::Module {
//     let conv1 = nn::conv2d(p, 13, 64, 3, Default::default());
//     let conv2 = nn::conv2d(p, 64, 64, 3, Default::default());
//     let conv3 = nn::conv2d(p, 64, 128, 3, Default::default());
//     let conv4 = nn::conv2d(p, 128, 128, 3, Default::default());
//     let conv5 = nn::conv2d(p, 128, 256, 3, Default::default());
//     let conv6 = nn::conv2d(p, 256, 256, 3, Default::default());
//     let conv7 = nn::conv2d(p, 256, 512, 3, Default::default());
//     let conv8 = nn::conv2d(p, 512, 512, 3, Default::default());
//     let conv9 = nn::conv2d(p, 512, 1024, 3, Default::default());
//     let conv10 = nn::conv2d(p, 1024, 2048, 3, Default::default());
//     let fc1 = nn::linear(p, 2048, 4096, Default::default());
//     let fc2 = nn::linear(p, 4096, 1, Default::default()); // Output layer for evaluating board position

//     move |xs: &Tensor| {
//         xs.apply(&conv1)
//             .relu()
//             .apply(&conv2)
//             .relu()
//             .apply(&conv3)
//             .relu()
//             .apply(&conv4)
//             .relu()
//             .apply(&conv5)
//             .relu()
//             .apply(&conv6)
//             .relu()
//             .apply(&conv7)
//             .relu()
//             .apply(&conv8)
//             .relu()
//             .apply(&conv9)
//             .relu()
//             .apply(&conv10)
//             .relu()
//             .flat_view()
//             .apply(&fc1)
//             .relu()
//             .apply(&fc2)
//     }
// }

  
    
    
    
    
// use rand;
// use std::error::Error;
// use std::path::Path;
// use std::result::Result;
// use tensorflow::Code;
// use tensorflow::SessionRunArgs;
// use tensorflow::Status;
// use tensorflow::{SavedModelBundle, Graph, ImportGraphDefOptions, Session, SessionOptions, Tensor};

// // #[cfg_attr(feature = "examples_system_alloc", global_allocator)]
// // #[cfg(feature = "examples_system_alloc")]
// // static ALLOCATOR: std::alloc::System = std::alloc::System;

// fn regress() -> Result<(), Box<dyn Error>> {
//     let export_dir = "models/saved_chess_model";
//     if !Path::new(export_dir).exists() {
//         return Err(Box::new(
//             Status::new_set(
//                 Code::NotFound,
//                 &format!(
//                     "Run 'python train.py' to generate {} and try again.",
//                     export_dir
//                 ),
//             )
//             .unwrap(),
//         ));
//     }

//     // Generate some test data.
//     let w = 0.1;
//     let b = 0.3;
//     let num_points = 100;
//     let steps = 201;
//     let mut x = Tensor::new(&[num_points as u64]);
//     let mut y = Tensor::new(&[num_points as u64]);
//     for i in 0..num_points {
//         x[i] = (2.0 * rand::random::<f64>() - 1.0) as f32;
//         y[i] = w * x[i] + b;
//     }

//     // Load the saved model exported by regression_savedmodel.py.
//     let mut graph = Graph::new();
//     let bundle =
//         SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;
//     let session = &bundle.session;

//     // train
//     let train_signature = bundle.meta_graph_def().get_signature("train")?;
//     let x_info = train_signature.get_input("x")?;
//     let y_info = train_signature.get_input("y")?;
//     let loss_info = train_signature.get_output("loss")?;
//     let op_x = graph.operation_by_name_required(&x_info.name().name)?;
//     let op_y = graph.operation_by_name_required(&y_info.name().name)?;
//     let op_train = graph.operation_by_name_required(&loss_info.name().name)?;

//     // internal parameters
//     let op_b = {
//         let b_signature = bundle.meta_graph_def().get_signature("b")?;
//         let b_info = b_signature.get_output("output")?;
//         graph.operation_by_name_required(&b_info.name().name)?
//     };

//     let op_w = {
//         let w_signature = bundle.meta_graph_def().get_signature("w")?;
//         let w_info = w_signature.get_output("output")?;
//         graph.operation_by_name_required(&w_info.name().name)?
//     };

//     // Train the model (e.g. for fine tuning).
//     let mut train_step = SessionRunArgs::new();
//     train_step.add_feed(&op_x, 0, &x);
//     train_step.add_feed(&op_y, 0, &y);
//     train_step.add_target(&op_train);
//     for _ in 0..steps {
//         session.run(&mut train_step)?;
//     }

//     // Grab the data out of the session.
//     let mut output_step = SessionRunArgs::new();
//     let w_ix = output_step.request_fetch(&op_w, 0);
//     let b_ix = output_step.request_fetch(&op_b, 0);
//     session.run(&mut output_step)?;

//     // Check our results.
//     let w_hat: f32 = output_step.fetch(w_ix)?[0];
//     let b_hat: f32 = output_step.fetch(b_ix)?[0];
//     println!(
//         "Checking w: expected {}, got {}. {}",
//         w,
//         w_hat,
//         if (w - w_hat).abs() < 1e-3 {
//             "Success!"
//         } else {
//             "FAIL"
//         }
//     );
//     println!(
//         "Checking b: expected {}, got {}. {}",
//         b,
//         b_hat,
//         if (b - b_hat).abs() < 1e-3 {
//             "Success!"
//         } else {
//             "FAIL"
//         }
//     );
//     Ok(())
// }

// fn infer() -> Result<(), Box<dyn std::error::Error>> {
//     let export_dir = "models/saved_chess_model";
//     if !Path::new(export_dir).exists() {
//         return Err(Box::new(
//             Status::new_set(
//                 Code::NotFound,
//                 &format!(
//                     "Run 'python train.py' to generate {} and try again.",
//                     export_dir
//                 ),
//             )
//             .unwrap(),
//         ));
//     }
//     // Load the saved TensorFlow model
//     // let mut graph = Graph::new();
//     // let model = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;
//     // let mut args = SessionRunArgs::new();
//     // let x = Tensor::new(&[1, 2]).with_values(&[1.0, 2.0])?;
//     // let y = Tensor::new(&[1, 2]).with_values(&[3.0, 4.0])?;
//     // let x_info = model.meta_graph_def().get_signature("serving_default")?.get_input("x")?;
//     // let y_info = model.meta_graph_def().get_signature("serving_default")?.get_input("y")?;
//     // let op_x = graph.operation_by_name_required(&x_info.name().name)?;
//     // let op_y = graph.operation_by_name_required(&y_info.name().name)?;
//     // args.add_feed(&op_x, 0, &x);
//     // args.add_feed(&op_y, 0, &y);
//     // let output_tensors = model.session.run(&mut SessionRunArgs::new())?;
//     let export_dir = Path::new("model");
//     let mut graph = Graph::new();
//     let model = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;

//     // Define the input tensors for the model
//     let x = Tensor::new(&[1, 2]).with_values(&[1.0, 2.0])?;
//     let y = Tensor::new(&[1, 2]).with_values(&[3.0, 4.0])?;
//     let x_info = model.meta_graph_def().get_signature("serving_default")?.get_input("x")?;
//     let y_info = model.meta_graph_def().get_signature("serving_default")?.get_input("y")?;
//     let op_x = graph.operation_by_name_required(&x_info.name().name)?;
//     let op_y = graph.operation_by_name_required(&y_info.name().name)?;

//     // Run the model and get the output tensors
//     let mut args = SessionRunArgs::new();
//     args.add_feed(&op_x, 0, &x);
//     args.add_feed(&op_y, 0, &y);
//     let output_tensors = model.session.run(&mut args)?;

//     // Import the model into the graph from a file
//     // let mut model_file = File::open("model.pb")?;
//     // let mut model_buffer = Vec::new();
//     // model_file.read_to_end(&mut model_buffer)?;
//     // let mut graph = Graph::new();
//     // let import_opts = ImportGraphDefOptions::new();
//     // graph.import_graph_def(&model_buffer, &import_opts)?;
//     // Create a new TensorFlow session
//     // let session = Session::new(&SessionOptions::new(), &graph)?;

//     // Define the input tensors for the model
//     // let input_tensor = Tensor::new(&[1, 28, 28, 1])
//     //     .with_values(&[0.0; 28 * 28 * 1])?;

//     // let mut args = SessionRunArgs::new();
//     // args.add_feed(&graph.operation_by_name_required("input")?, 0, &input_tensor);

//     // // Run the model and get the output tensors
//     // let output_tensors = session.run(&mut SessionRunArgs::new())?;

//     // Print the output tensors
//     println!("Output tensors: {:?}", output_tensors);

//     Ok(())
// }
// // Note: You'll need to install the tensorflow crate and its dependencies in order to run this code. Also, this code assumes that you have a saved TensorFlow model named model.pb in the current directory.