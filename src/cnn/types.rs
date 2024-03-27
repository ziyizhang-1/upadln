extern crate openvino;
use openvino::{Core, Blob, TensorDesc};
use std::time::Instant;
use crate::utils::openvino_tensor_converter::{convert, Dimensions, Precision};

static MODEL_XML_PATH: &'static str = "/home/ziyi/tmp/yolov8n_openvino_model_fp16/yolov8n.xml";
static MODEL_WEIGHTS_PATH: &'static str = "/home/ziyi/tmp/yolov8n_openvino_model_fp16/yolov8n.bin";

type Error = Box<dyn std::error::Error + 'static>;

pub fn run() -> Result<(), Error> {
    let input_shape = TensorDesc::new(openvino::Layout::NCHW, &[1, 3, 640, 640], openvino::Precision::FP32);
    let mut core = Core::new(Some(MODEL_XML_PATH))?;
    let cnn = core.read_network_from_file(MODEL_XML_PATH, MODEL_WEIGHTS_PATH)?;
    let mut model = core.load_network(&cnn, "CPU")?;
    let mut irq = model.create_infer_request()?;
    
    let tensors = convert("/home/ziyi/tmp/bus.jpg", &Dimensions::new(640, 640, 3, Precision::FP32))?;
    let input = Blob::new(&input_shape, &tensors)?;
    // irq.set_batch_size(1)?;

    irq.set_blob("x", &input).unwrap();
    // irq.set_blob("Result_3426", &output).unwrap();

    let now = Instant::now();
    (0..1000).into_iter().for_each(|_| {
        irq.infer().unwrap();
    });
    // irq.infer().unwrap();
    let output = irq.get_blob("__module.model.22/aten::cat/Concat_5").unwrap();
    // println!("{:#?}", unsafe { output.buffer_as_type::<f32>().unwrap() });
    println!("{:#?}", now.elapsed());

    Ok(())
}
