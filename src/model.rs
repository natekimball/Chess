use std::{sync::Arc};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

#[derive(Clone)]
pub struct Model {
    pred_input_parameter_name: String,
    pred_output_parameter_name: String,
    graph: Arc<Graph>,
    bundle: Arc<SavedModelBundle>,
}

impl Model {
    pub fn new() -> Self {
        let pred_input_parameter_name = "conv2d_input".to_owned();
        let pred_output_parameter_name = "dense_4".to_owned();
        let save_dir = "model/saved_model";
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, save_dir
        ).expect("Can't load saved model");
        // println!("sigs: {:?}", bundle.meta_graph_def().signatures());
        Self {
            pred_input_parameter_name,
            pred_output_parameter_name,
            graph: Arc::new(graph),
            bundle: Arc::new(bundle),
        }
    }

    pub fn run_inference(&self, input_data: [[[u8; 8]; 8]; 13]) -> Result<f32, Box<dyn std::error::Error>> {
        let input_tensor: Tensor<f32> = Tensor::new(&[1,13,8,8]).with_values(&input_data.into_iter().flatten().flatten().map(|u| u as f32).collect::<Vec<f32>>()).unwrap();

        let session = &self.bundle.session;

        let signature_train = self.bundle.meta_graph_def().get_signature("serving_default").unwrap();
        let input_info_pred = signature_train.get_input(self.pred_input_parameter_name.as_str()).unwrap();
        let output_info_pred = signature_train.get_output(self.pred_output_parameter_name.as_str()).unwrap();
        let input_op_pred = self.graph.operation_by_name_required(&input_info_pred.name().name).unwrap();
        let output_op_pred = self.graph.operation_by_name_required(&output_info_pred.name().name).unwrap();
    
        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op_pred, 0, &input_tensor);
    
        let out = args.request_fetch(&output_op_pred, 0);
    
        session
        .run(&mut args)
        .expect("Error occurred during inference");
    
        let prediction = args.fetch(out)?;
    
        // println!("data : {:?}", input_tensor);
        println!("Prediction: {:?}", prediction);
        
        Ok(prediction[0])
    }
}

// https://github.com/Grimmp/RustTensorFlowTraining
