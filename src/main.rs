use dtw::ucr::*;

// Working main function to try with naive implementation
fn main() {
    // Initialize logger
    pretty_env_logger::init();

    // Input parameters
    let data_name = "Data.txt";
    let query_name = "Query2.txt";

    //let settings = Settings::default();
    let settings = Settings::new(
        true,   // jump
        true,   // sort:
        true,   // normalize:
        0.10,   // window_rate:
        100000, // epoch:
    );
    println!("{:?}", settings);
    let mut trillion = Trillion::new(data_name, query_name, settings);
    trillion.calculate();
    trillion.print();
}
