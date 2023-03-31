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
        let pred_input_parameter_name = "input_1".to_owned();
        let pred_output_parameter_name = "value".to_owned();
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
            bundle: Arc::new(bundle)
        }
    }

    pub fn run_inference(&mut self, input_data: &Vec<[[[f32; 8]; 8]; 13]>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {

        let len = input_data.len() as u64;
        let now = std::time::SystemTime::now();

        let data = input_data.clone().into_iter().flatten().flatten().flatten().collect::<Vec<f32>>();
        let input_tensor: Tensor<f32> = Tensor::new(&[len,13,8,8]).with_values(&data)?;

        let session = &self.bundle.session;

        let signature_train = self.bundle.meta_graph_def().get_signature("serving_default")?;
        let input_info_pred = signature_train.get_input(self.pred_input_parameter_name.as_str())?;
        let output_info_pred = signature_train.get_output(self.pred_output_parameter_name.as_str())?;
        let input_op_pred = self.graph.operation_by_name_required(&input_info_pred.name().name)?;
        let output_op_pred = self.graph.operation_by_name_required(&output_info_pred.name().name)?;
    
        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op_pred, 0, &input_tensor);
    
        let out = args.request_fetch(&output_op_pred, 0);
    
        session
        .run(&mut args)
        .expect("Error occurred during inference");
    
        let prediction = args.fetch(out)?;
        // println!("data : {:?}", input_tensor);
        let elapsed = now.elapsed().unwrap();
        println!("time to inference {len} inputs: {:?}, {}ms/inf", elapsed, elapsed.as_millis() / len as u128);
        // println!("Prediction: {:?}", prediction);
        
        Ok(prediction.to_vec())
    }
}