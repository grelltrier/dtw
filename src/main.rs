use dtw::ucr_fixed::*;

// Working main function to try with naive implementation
fn main() {
    // Initialize logger
    pretty_env_logger::init();

    // Input parameters
    let data_name = "Data.txt";
    let query_name = "Query2.txt";
    // let window_rate = 0.10;
    // let sort = true;
    // let dont_jump = false;

    let settings = Settings::default();
    Trillion::calculate(data_name, query_name, settings);
}
