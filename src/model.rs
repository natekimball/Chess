use std::{sync::Arc, path::Path, error::Error, time::Duration};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Session, Tensor, Operation, SavedModelBuilder, SavedModelSaver, Scope, ImportGraphDefOptions, MetaGraphDef, SignatureDef, REGRESS_INPUTS, REGRESS_METHOD_NAME, REGRESS_OUTPUTS, Output, Status, Variable, ops, DataType};

// const REGRESS_METHOD_NAME: &str = "serve";

#[derive(Clone)]
pub struct Model {
    session: Arc<Session>,
    input_op_train: Operation,
    target_op_train: Operation,
    output_op_train: Operation,
    input_op_pred: Operation,
    output_op_pred: Operation,
    save_op: Operation
}

impl Model {
    pub fn new(save_dir: &str) -> Self {
        let input_parameter_name = "input";
        let output_parameter_name = "output_0";
        let train_input_target_name = "training_target";
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(), &["serve"], &mut graph, save_dir
        ).expect("Can't load saved model");
        // println!("sigs: {:?}", bundle.meta_graph_def().signatures());
        
        let signature_train = bundle.meta_graph_def().get_signature("train").expect("signature error").clone();
        let input_info_train = signature_train.get_input(input_parameter_name).expect("signature error");
        let target_info_train = signature_train.get_input(train_input_target_name).expect("signature error");
        let output_info_train = signature_train.get_output(output_parameter_name).expect("signature error");  
        let input_op_train = graph.operation_by_name_required(&input_info_train.name().name).expect("operation error");
        let target_op_train = graph.operation_by_name_required(&target_info_train.name().name).expect("operation error");
        let output_op_train = graph.operation_by_name_required(&output_info_train.name().name).expect("operation error");
        

        let signature_pred = bundle.meta_graph_def().get_signature("pred").expect("signature error").clone();
        let input_info_pred = signature_pred.get_input(input_parameter_name).expect("signature error");
        let output_info_pred = signature_pred.get_output(output_parameter_name).expect("signature error");
        let input_op_pred = graph.operation_by_name_required(&input_info_pred.name().name).expect("operation error");
        let output_op_pred = graph.operation_by_name_required(&output_info_pred.name().name).expect("operation error");

        let signature_save = bundle.meta_graph_def().get_signature("save").expect("signature error").clone();
        let output_info_save = signature_save.get_output(output_parameter_name).expect("signature error");
        let save_op = graph.operation_by_name_required(&output_info_save.name().name).expect("operation error");
        
        Self {
            session: Arc::new(bundle.session),
            input_op_train,
            target_op_train,
            output_op_train,
            input_op_pred,
            output_op_pred,
            save_op
        }
    }

    pub fn run_inference(&self, input_data: &Vec<[[[f32; 8]; 8]; 13]>) -> Result<Vec<f32>, Box<dyn Error>> {
        let len = input_data.len() as u64;
        let now = std::time::SystemTime::now();

        let data = input_data.clone().into_iter().flatten().flatten().flatten().collect::<Vec<f32>>();
        let input_tensor: Tensor<f32> = Tensor::new(&[len,13,8,8]).with_values(&data)?;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input_op_pred, 0, &input_tensor);
    
        let out = args.request_fetch(&self.output_op_pred, 0);
    
        self.session
        .run(&mut args)
        .expect("Error occurred during inference");
    
        let prediction = args.fetch(out)?;
        // println!("data : {:?}", input_tensor);
        let elapsed = now.elapsed().unwrap_or(Duration::from_secs(0));
        println!("time to inference {len} inputs: {:?}, {}ms/inf", elapsed, elapsed.as_millis() / len as u128);
        // println!("Prediction: {:?}", prediction);
        
        Ok(prediction.to_vec())
        // Err("this method shouldn't have been called".to_string())
    }

    pub fn back_propagate(&self, input_data: &Vec<[[[f32; 8]; 8]; 13]>, amplified_scores: &Vec<f32>) {
        let len = input_data.len() as u64;
        let now = std::time::SystemTime::now();

        let data = input_data.clone().into_iter().flatten().flatten().flatten().collect::<Vec<f32>>();
        let input_tensor: Tensor<f32> = Tensor::new(&[len,13,8,8]).with_values(&data).expect("Can't create tensor from input data");
        let target_tensor: Tensor<f32> = Tensor::new(&[len,1]).with_values(&amplified_scores).expect("Can't create tensor from target data");

        let mut args = SessionRunArgs::new();

        args.add_feed(&self.input_op_train, 0, &input_tensor);
        args.add_feed(&self.target_op_train, 0, &target_tensor);

        let out = args.request_fetch(&self.output_op_train, 0);

        self.session
        .run(&mut args)
        .expect("Error occurred during calculations");
        
        let loss: f32 = args.fetch(out).unwrap()[0];

        let elapsed = now.elapsed().unwrap();
        println!("time to train on {len} inputs: {:?}, {}ms/input", elapsed, elapsed.as_millis() / len as u128);
        println!("Loss: {:?}", loss);
    }

    pub fn save_model(&self) {
        let mut args = SessionRunArgs::new();
        args.add_target(
            &self.save_op
        );
        self.session.run(&mut args).expect("Error occurred during saving");
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_and_save_model() {
        let model = Model::new( "model/saved_model_t");
        model.save_model();
    }
}