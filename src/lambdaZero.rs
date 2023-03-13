use rand;
use std::error::Error;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::{SavedModelBundle, Graph, ImportGraphDefOptions, Session, SessionOptions, Tensor};

// #[cfg_attr(feature = "examples_system_alloc", global_allocator)]
// #[cfg(feature = "examples_system_alloc")]
// static ALLOCATOR: std::alloc::System = std::alloc::System;

fn regress() -> Result<(), Box<dyn Error>> {
    let export_dir = "models/saved_chess_model";
    if !Path::new(export_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python train.py' to generate {} and try again.",
                    export_dir
                ),
            )
            .unwrap(),
        ));
    }

    // Generate some test data.
    let w = 0.1;
    let b = 0.3;
    let num_points = 100;
    let steps = 201;
    let mut x = Tensor::new(&[num_points as u64]);
    let mut y = Tensor::new(&[num_points as u64]);
    for i in 0..num_points {
        x[i] = (2.0 * rand::random::<f64>() - 1.0) as f32;
        y[i] = w * x[i] + b;
    }

    // Load the saved model exported by regression_savedmodel.py.
    let mut graph = Graph::new();
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;
    let session = &bundle.session;

    // train
    let train_signature = bundle.meta_graph_def().get_signature("train")?;
    let x_info = train_signature.get_input("x")?;
    let y_info = train_signature.get_input("y")?;
    let loss_info = train_signature.get_output("loss")?;
    let op_x = graph.operation_by_name_required(&x_info.name().name)?;
    let op_y = graph.operation_by_name_required(&y_info.name().name)?;
    let op_train = graph.operation_by_name_required(&loss_info.name().name)?;

    // internal parameters
    let op_b = {
        let b_signature = bundle.meta_graph_def().get_signature("b")?;
        let b_info = b_signature.get_output("output")?;
        graph.operation_by_name_required(&b_info.name().name)?
    };

    let op_w = {
        let w_signature = bundle.meta_graph_def().get_signature("w")?;
        let w_info = w_signature.get_output("output")?;
        graph.operation_by_name_required(&w_info.name().name)?
    };

    // Train the model (e.g. for fine tuning).
    let mut train_step = SessionRunArgs::new();
    train_step.add_feed(&op_x, 0, &x);
    train_step.add_feed(&op_y, 0, &y);
    train_step.add_target(&op_train);
    for _ in 0..steps {
        session.run(&mut train_step)?;
    }

    // Grab the data out of the session.
    let mut output_step = SessionRunArgs::new();
    let w_ix = output_step.request_fetch(&op_w, 0);
    let b_ix = output_step.request_fetch(&op_b, 0);
    session.run(&mut output_step)?;

    // Check our results.
    let w_hat: f32 = output_step.fetch(w_ix)?[0];
    let b_hat: f32 = output_step.fetch(b_ix)?[0];
    println!(
        "Checking w: expected {}, got {}. {}",
        w,
        w_hat,
        if (w - w_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    println!(
        "Checking b: expected {}, got {}. {}",
        b,
        b_hat,
        if (b - b_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    Ok(())
}

fn infer() -> Result<(), Box<dyn std::error::Error>> {
    let export_dir = "models/saved_chess_model";
    if !Path::new(export_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python train.py' to generate {} and try again.",
                    export_dir
                ),
            )
            .unwrap(),
        ));
    }
    // Load the saved TensorFlow model
    // let mut graph = Graph::new();
    // let model = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;
    // let mut args = SessionRunArgs::new();
    // let x = Tensor::new(&[1, 2]).with_values(&[1.0, 2.0])?;
    // let y = Tensor::new(&[1, 2]).with_values(&[3.0, 4.0])?;
    // let x_info = model.meta_graph_def().get_signature("serving_default")?.get_input("x")?;
    // let y_info = model.meta_graph_def().get_signature("serving_default")?.get_input("y")?;
    // let op_x = graph.operation_by_name_required(&x_info.name().name)?;
    // let op_y = graph.operation_by_name_required(&y_info.name().name)?;
    // args.add_feed(&op_x, 0, &x);
    // args.add_feed(&op_y, 0, &y);
    // let output_tensors = model.session.run(&mut SessionRunArgs::new())?;
    let export_dir = Path::new("model");
    let mut graph = Graph::new();
    let model = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;

    // Define the input tensors for the model
    let x = Tensor::new(&[1, 2]).with_values(&[1.0, 2.0])?;
    let y = Tensor::new(&[1, 2]).with_values(&[3.0, 4.0])?;
    let x_info = model.meta_graph_def().get_signature("serving_default")?.get_input("x")?;
    let y_info = model.meta_graph_def().get_signature("serving_default")?.get_input("y")?;
    let op_x = graph.operation_by_name_required(&x_info.name().name)?;
    let op_y = graph.operation_by_name_required(&y_info.name().name)?;

    // Run the model and get the output tensors
    let mut args = SessionRunArgs::new();
    args.add_feed(&op_x, 0, &x);
    args.add_feed(&op_y, 0, &y);
    let output_tensors = model.session.run(&mut args)?;

    // Import the model into the graph from a file
    // let mut model_file = File::open("model.pb")?;
    // let mut model_buffer = Vec::new();
    // model_file.read_to_end(&mut model_buffer)?;
    // let mut graph = Graph::new();
    // let import_opts = ImportGraphDefOptions::new();
    // graph.import_graph_def(&model_buffer, &import_opts)?;
    // Create a new TensorFlow session
    // let session = Session::new(&SessionOptions::new(), &graph)?;

    // Define the input tensors for the model
    // let input_tensor = Tensor::new(&[1, 28, 28, 1])
    //     .with_values(&[0.0; 28 * 28 * 1])?;

    // let mut args = SessionRunArgs::new();
    // args.add_feed(&graph.operation_by_name_required("input")?, 0, &input_tensor);

    // // Run the model and get the output tensors
    // let output_tensors = session.run(&mut SessionRunArgs::new())?;

    // Print the output tensors
    println!("Output tensors: {:?}", output_tensors);

    Ok(())
}
// Note: You'll need to install the tensorflow crate and its dependencies in order to run this code. Also, this code assumes that you have a saved TensorFlow model named model.pb in the current directory.