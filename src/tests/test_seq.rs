// Create random sequences for the query and the data time series
// The observations are of type f64
// The tuple "query_rng" describes the min and max length of the query
// If a tuple "data_rng" was provided, it describes the min and max length of the data sequence
// Otherwise the data sequence has the same length as the query
pub fn make_rdm_series(
    query_rng: (u32, u32),
    data_rng: Option<(u32, u32)>,
) -> (std::vec::Vec<f64>, std::vec::Vec<f64>) {
    use rand::thread_rng;
    use rand::Rng;

    let mut rng = thread_rng();
    let query_len: u32 = rng.gen_range(query_rng.0..query_rng.1);
    let data_len: u32;
    if let Some((start, end)) = data_rng {
        data_len = rng.gen_range(start..end)
    } else {
        data_len = query_len;
    };

    let mut query = Vec::new();
    let mut data = Vec::new();

    for _ in 0..data_len {
        let observation: f64 = rand::random();
        data.push(observation);
    }
    for _ in 0..query_len {
        let observation: f64 = rand::random();
        query.push(observation);
    }

    (query, data)
}

pub fn make_rdm_params(
    query_rng: (u32, u32),
    data_rng: Option<(u32, u32)>,
) -> (
    std::vec::Vec<f64>,
    std::vec::Vec<f64>,
    std::vec::Vec<f64>,
    std::vec::Vec<f64>,
    usize,
    f64,
) {
    use rand::thread_rng;
    use rand::Rng;

    let (query, data) = make_rdm_series(query_rng, data_rng);

    let (mut cb_query, mut cb_data) = (Vec::new(), Vec::new());

    let mut rng = thread_rng();
    let mut observation: f64;
    for i in 0..query.len() {
        observation = rng.gen_range(0.0..0.1);
        if i == 0 {
            cb_query.push(observation);
        } else {
            cb_query.push(observation + cb_query[i - 1]);
        }
    }
    for i in 0..data.len() {
        observation = rand::random();
        if i == 0 {
            cb_data.push(observation);
        } else {
            cb_data.push(observation + cb_data[i - 1]);
        }
    }
    let bsf = rng.gen_range(1000.0..2000.0);

    let w = rng.gen_range(0..query.len() - 2);

    (query, data, cb_query, cb_data, w, bsf)
}

pub fn make_test_series(equal_len: bool) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let a1 = vec![1.0, 1.];
    let a2 = vec![2.0, 1.];
    let a3 = vec![3.0, 1.];
    let a4 = vec![2.0, 1.];
    let a5 = vec![2.13, 1.];
    let a6 = vec![1.0, 1.];

    let b1 = vec![1.0, 1.];
    let b2 = vec![1.0, 1.];
    let b3 = vec![2.0, 1.];
    let b4 = vec![2.0, 1.];
    let b5 = vec![2.42, 1.];
    let b6 = vec![3.0, 1.];
    let b7 = vec![2.0, 1.];
    let b8 = vec![1.0, 1.];
    let mut data = Vec::new();
    data.push(a1);
    data.push(a2);
    data.push(a3);
    data.push(a4);
    data.push(a5);
    data.push(a6);

    let mut query = Vec::new();
    query.push(b1);
    query.push(b2);
    query.push(b3);
    query.push(b4);
    query.push(b5);
    query.push(b6);

    if !equal_len {
        query.push(b7);
        query.push(b8);
    }
    (query, data)
}

pub fn make_knn_fail_candidate(candidate_no: usize) -> Vec<f64> {
    let failing_candidate_1 = vec![
        -0.6201428213355223,
        -0.5728859410962197,
        -0.6234087792265741,
        -0.4270118151567179,
        -0.586208683405387,
        0.0045554996676052525,
        -0.03235286276777742,
        -0.001523333405307244,
        0.28714126252566113,
        0.30318264937729583,
        0.27730045047275065,
        0.05206894917342068,
        0.13173571709892984,
        -0.2298314989596042,
        -0.03653719001549565,
        0.40279430552296774,
        0.21560166806399333,
        0.44777325952651,
        0.7871026540733933,
        0.3558437525231183,
        -0.03407735230862473,
        0.12047364684090218,
        0.012265709750020415,
        0.19897711696853243,
        0.41968231015088164,
        0.6123228018958196,
        0.9614608885959601,
        1.142382379480056,
        1.464619955511758,
        1.139238309437359,
        1.133883297085439,
        1.0914760186533556,
        0.6574148114518049,
        0.7270406335604327,
        0.44116191802455695,
        0.8240953616875615,
        0.6062400722217697,
        0.7493165281123629,
        0.808664090570953,
        0.5592000056346818,
        -0.02817865585912427,
        -0.044194714803366265,
        -0.31766947194180684,
        -0.15139743703973296,
        -0.014004146848230628,
        0.44396281656986625,
        0.6039622923791491,
        0.4298071681908666,
        0.5327253880586091,
        0.25966145715480354,
        0.2543819143901895,
        0.24133349604825916,
        0.24134518585167117,
        0.15533329004913407,
        0.4516383360782669,
        -0.05545835452574098,
        0.060406863815440366,
        0.30276422477044307,
        0.5005592974462986,
        0.6569259503674485,
        0.6678348099682139,
        0.8510533201630459,
        1.0049961789378454,
        0.935819223632456,
        0.8336771309440286,
        0.753610880222751,
        0.35444346560884427,
        0.29112261473774576,
        0.3231734579594731,
        0.4083599508141456,
        0.7989693740009628,
        0.7039965673898426,
        0.8726419516893764,
        1.0888623463807499,
        1.3434646411112157,
        1.075007277483928,
        1.1323834303643214,
        1.1967536846292222,
        0.9240556283363753,
        0.7232603080834131,
        1.016126414190726,
        0.9805429502615016,
        1.0860435613538308,
        1.1098526331546568,
        0.937897437965438,
        0.7864784131592443,
        0.9065298529292523,
        0.6494884482451874,
        0.8608740809559369,
        1.0148333108674594,
        0.7924796389261712,
        0.7206071286048312,
        0.39919787797184686,
        -0.19674415983534238,
        0.07015607868189298,
        -0.07018517968812342,
        0.01839960937372795,
        0.0817348019017904,
        0.08754349768891015,
        -0.18412117457036603,
        -0.44041634766495263,
        -0.5417358759712049,
        -0.8626334893449906,
        -1.1483579644191797,
        -0.7499088719891049,
        -0.7348247393263198,
        -1.0642275587116488,
        -1.075383500708144,
        -1.380710314878901,
        -1.7458211649550912,
        -1.8164745803150437,
        -1.5584693892232782,
        -1.5236585617681835,
        -1.3460202279396039,
        -1.6620289788143359,
        -1.7866675028551753,
        -1.85768305858333,
        -2.185958715734383,
        -2.542994807043975,
        -2.291009864709411,
        -2.2879670251170947,
        -2.462541737480625,
        -2.2445136986977445,
        -2.1818363282089104,
        -2.4496626318700585,
        -2.0871743567176844,
        -2.0088358968243285,
        -1.608644920858885,
    ];

    let failing_candidate_2 = vec![
        -0.6035390967355747,
        -0.409527508019152,
        -0.56679082150203,
        0.01679812681788141,
        -0.01966195833121064,
        0.010793125327134327,
        0.2959516924566714,
        0.3117982457017277,
        0.2862304037917113,
        0.06373449300703483,
        0.14243365358893137,
        -0.21474208152242863,
        -0.0237954640647844,
        0.4102000502052935,
        0.22528099526249293,
        0.45463270407221495,
        0.7898407112123166,
        0.3638197437593372,
        -0.021365502768005685,
        0.13130836887109457,
        0.024414691129408552,
        0.20885836070190733,
        0.42688293847512676,
        0.6171836797903266,
        0.9620812458138684,
        1.1408053215428715,
        1.459129101984827,
        1.1376994383796144,
        1.1324094663125177,
        1.0905172532596634,
        0.6617280160727406,
        0.7305081849870482,
        0.44810166183123357,
        0.8263841171103533,
        0.6111748290253262,
        0.7525135235754913,
        0.8111402693277461,
        0.5647060961958252,
        -0.015538450018267405,
        -0.03135998298067628,
        -0.30151320229110734,
        -0.13726065488925557,
        -0.0015361001066843856,
        0.45086854154947453,
        0.6089247143756115,
        0.4368848231945211,
        0.5385530309439274,
        0.2688056481027381,
        0.2635902289932312,
        0.2507002926098258,
        0.2507118404325861,
        0.16574461788821857,
        0.45845083662771735,
        -0.04248681810064303,
        0.07197113797729406,
        0.31138490314769174,
        0.5067776194855758,
        0.661245092540359,
        0.672021456585763,
        0.8530146527561068,
        1.0050877703009558,
        0.9367510163957895,
        0.8358495092306721,
        0.7567557178445828,
        0.3624364642866439,
        0.2998846884251399,
        0.3315462522512312,
        0.41569809693064624,
        0.8015633017162193,
        0.70774400474925,
        0.8743410755980967,
        1.0879353258230868,
        1.339445301679613,
        1.0742485362070737,
        1.1309278164977916,
        1.1945162500316795,
        0.9251302980042473,
        0.7267737741465584,
        1.016082821239884,
        0.9809315427991093,
        1.0851507768667799,
        1.1086706712186019,
        0.9388039893953704,
        0.7892240521306988,
        0.9078173851171939,
        0.6538979239644314,
        0.8627161336973042,
        1.0148054235402322,
        0.7951523889092312,
        0.7241528193460229,
        0.4066473037247099,
        -0.18205661074491802,
        0.08160194196622673,
        -0.057034775906582146,
        0.030474090350056788,
        0.09304003367929929,
        0.0987781788862112,
        -0.16958694026299057,
        -0.4227692333189308,
        -0.5228581667084649,
        -0.8398582592547637,
        -1.1221124153047124,
        -0.7285027594167552,
        -0.7136018338542027,
        -1.0390038308722687,
        -1.0500242763288412,
        -1.3516426876700807,
        -1.7123190169421043,
        -1.782114298285405,
        -1.5272427565695497,
        -1.4928547304048787,
        -1.3173739355451244,
        -1.6295445441686462,
        -1.7526692481525237,
        -1.8228222714217548,
        -2.1471107963612237,
        -2.499810440330452,
        -2.2508860273361866,
        -2.2478801451103054,
        -2.4203345282814936,
        -2.204954589037897,
        -2.143038478049637,
        -2.4076118482106152,
        -2.0495262408975945,
        -1.972139255138987,
        -1.5768088720816649,
        -1.2726017970739052,
        -1.4554793733777427,
    ];
    match candidate_no {
        2 => failing_candidate_2,
        _ => failing_candidate_1,
    }
}

pub fn make_knn_fail_query() -> Vec<f64> {
    vec![
        0.3340897059175602,
        1.330351738237077,
        1.8294226783075387,
        2.2471324255486183,
        1.7824557537475956,
        1.9521980391051645,
        1.6560067460872607,
        1.4907844168136324,
        1.1883776417051173,
        0.6441962410552153,
        0.16208037903720438,
        0.05072376217421901,
        -0.8505728539771462,
        -0.5282231450842175,
        -0.5907185503934304,
        -0.219793737318905,
        -0.07673732062660284,
        0.18549628273893906,
        -0.6013423886946005,
        -1.1426222044037317,
        -1.3797660079451424,
        -1.3215889482166503,
        -0.7227615586581437,
        -0.8765953526130663,
        -0.20730614698617456,
        0.04492048963428961,
        0.6678234428174481,
        0.7351725403256216,
        0.3867126607446516,
        -0.3992124658283756,
        -0.31672104867348166,
        0.11754416740080008,
        -0.03768231287147581,
        -0.3245851744850972,
        -0.4883285994632444,
        -1.070103116428511,
        -1.3316591069364476,
        -1.4275465718960638,
        -1.4032265372062223,
        -1.43706533030291,
        -1.1127342982453414,
        -1.054738094545063,
        -0.09235034250763519,
        0.07318733837678339,
        1.0305235640780377,
        0.6468866688872117,
        0.92623512753647,
        0.7881645054122534,
        1.1065574936666636,
        1.421654988123975,
        0.2619071891331608,
        -0.44212764878737765,
        -1.2066271113871787,
        -0.9934369634670003,
        -0.21345258929881888,
        -0.3868370061330327,
        0.04412388245461167,
        0.33354871276551656,
        -0.22437269533123347,
        -0.01350493864625855,
        -0.41240688141471143,
        0.392016636547235,
        0.3747391599025138,
        1.2426843762779887,
        1.0171785322128883,
        1.3298900103230242,
        1.2965774458917823,
        0.2238696532962006,
        -0.29717957544851453,
        0.027775413909104686,
        -0.0013432682110865728,
        -0.5948630074208062,
        -0.9271749697316352,
        -0.7947985793339815,
        -1.3216239320596148,
        -0.8043845915349692,
        -1.1442995474918198,
        -0.18464552093047823,
        -0.7575458528372373,
        -0.6518700818378529,
        -1.4588187400445871,
        -1.8427260170546214,
        -2.157462496818613,
        -1.9445419410265197,
        -1.4447129196806114,
        -1.2852964081902885,
        -1.4832212011988135,
        -1.0505809213190622,
        -0.6267482266224356,
        -0.5629758419566289,
        -0.25991242420423066,
        -0.04091218120619331,
        -0.5645240545882448,
        -0.1615510514735404,
        -0.5102153810841729,
        -0.8306116321831569,
        -0.7367362164332821,
        -0.8998757161873663,
        -0.9698102325090965,
        -0.6539557378084081,
        -0.09858061689040505,
        -0.20360340674936978,
        -0.029760647303379434,
        -0.1561864995604811,
        -0.03437869582489603,
        0.1990529403047628,
        -0.12821230681340653,
        0.01759657121584611,
        0.3364986145710296,
        0.38547065704967454,
        1.3032016322512974,
        0.9803442766886256,
        0.5893127137039592,
        -0.33904895584571676,
        0.14398843116241844,
        0.6040008466205529,
        0.5616153808445761,
        1.0382829238595062,
        1.1357429120658042,
        1.2900155033950864,
        1.3499359480095976,
        1.5833433305032,
        1.637278202251001,
        3.1160402952946313,
        2.4970906961519357,
        1.5133313100237922,
        0.908166914957359,
        0.3281165350707758,
    ]
}

pub fn make_knn_fail_cb1() -> Vec<f64> {
    vec![
        150.12604580558113,
        150.12509055943715,
        149.07001422232813,
        146.74060559641913,
        143.33902249543937,
        141.4355567837221,
        139.1722628663118,
        138.41726854519337,
        137.92210052198016,
        137.7610789062813,
        137.7610789062813,
        137.7610789062813,
        137.7610789062813,
        137.70947538942403,
        137.70947538942403,
        137.70947538942403,
        137.70947538942403,
        137.70947538942403,
        137.70947538942403,
        137.57145504823234,
        136.73826817628714,
        135.41591880133132,
        134.22398447332307,
        133.98100442956866,
        133.56270094717598,
        133.56270094717598,
        133.56270094717598,
        133.56270094717598,
        133.56270094717598,
        133.56270094717598,
        133.4366633436331,
        133.4366633436331,
        133.4366633436331,
        133.4366633436331,
        133.43661551669146,
        133.40749097888508,
        132.84133458952954,
        131.81315960965298,
        130.5813324326501,
        129.4028982907046,
        128.1498512029886,
        127.51772312496354,
        126.97445297053727,
        126.97445297053727,
        126.97445297053727,
        126.9318403677951,
        126.9318403677951,
        126.92140783602743,
        126.92140783602743,
        126.83266735641948,
        126.50708109293969,
        126.50708109293969,
        126.49159125515597,
        125.70134557042768,
        125.24468386782543,
        125.24468386782543,
        125.18925207713059,
        125.18925207713059,
        125.18925207713059,
        125.16072002260084,
        125.16072002260084,
        125.03330777175263,
        125.03330777175263,
        125.03330777175263,
        125.03330777175263,
        125.03330777175263,
        125.03330777175263,
        125.03330777175263,
        125.03330777175263,
        124.90543971020716,
        124.83608796202289,
        124.75055166933384,
        123.96558114666217,
        122.48133214233825,
        121.30210730260409,
        118.70115587839727,
        117.50101983940182,
        115.44058305558202,
        115.21422773666481,
        114.11452218177875,
        113.22528695598383,
        110.0497907622365,
        107.34053448794147,
        103.49611809095903,
        100.44132100702222,
        98.88389498147248,
        97.69894898407387,
        96.04392580611855,
        95.31488859085755,
        95.28016902174173,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.27971788558666,
        95.26728352979417,
        95.26728352979417,
        95.26728352979417,
        95.20530487957238,
        95.11654428728761,
        93.63871958715991,
        92.2827397999841,
        91.2223978601564,
        91.18131587256644,
        90.40900328394197,
        88.61654933435598,
        86.93579234916726,
        83.79188156391896,
        78.9520114916262,
        73.35689904301364,
        66.33104075573351,
        58.01334857829763,
        49.38164667340985,
        29.87407290196481,
        15.450876842881303,
        7.532118049964765,
        2.6530355366550356,
    ]
}

pub fn make_knn_fail_cb2() -> Vec<f64> {
    vec![
        151.50090407321818,
        151.50040716401963,
        150.6537280350319,
        148.63953516691444,
        145.42647991542145,
        144.44119529275414,
        143.09012073504795,
        142.33987713507696,
        141.84855505665485,
        141.68972337168822,
        141.68972337168822,
        141.68972337168822,
        141.68972337168822,
        141.62869769447133,
        141.62869769447133,
        141.62812515826144,
        141.62809963903516,
        141.62809963903516,
        141.62809963903516,
        141.47863984152954,
        140.6176783190913,
        139.2603975699539,
        138.03528758364308,
        137.77720379449383,
        137.33915404204046,
        137.3054778712942,
        137.3054778712942,
        137.3054778712942,
        137.3054778712942,
        137.3054778712942,
        137.29593272519847,
        137.29570144660687,
        137.29570144660687,
        137.29570144660687,
        137.29516913070594,
        137.26026913808536,
        136.66953868197163,
        135.60833809711406,
        134.34038694765016,
        133.12661467532038,
        131.83713603988824,
        131.17905637336705,
        130.61170863505606,
        130.61170863505606,
        130.61170863505606,
        130.5700357212478,
        130.5700357212478,
        130.5567888948617,
        130.5567888948617,
        130.5464927901018,
        130.372964543137,
        130.372964543137,
        130.35319212057354,
        129.53396093213436,
        128.80092306076455,
        128.77169376584317,
        128.65311671384524,
        128.65311671384524,
        128.65311671384524,
        128.6200342415093,
        128.6200342415093,
        128.483193388267,
        128.483193388267,
        128.483193388267,
        128.483193388267,
        128.483193388267,
        128.483193388267,
        128.483193388267,
        128.47741510270137,
        128.1209293675064,
        128.04688591022875,
        127.95614762836952,
        127.15557418914788,
        125.64989878447206,
        124.45156732776027,
        121.82227712145391,
        120.60286647879029,
        118.51719837151924,
        118.28242884774104,
        117.09630723437502,
        116.87558253675462,
        115.2454610019408,
        112.48763812488778,
        108.58540971015319,
        105.47905517069535,
        103.88475421620024,
        102.66761616553293,
        101.54305778935262,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.26456648361882,
        101.25451145584127,
        101.25451145584127,
        101.25451145584127,
        101.19800045029909,
        100.88991151396458,
        97.91093608516734,
        95.65131849933235,
        93.95373218109366,
        93.81344232266922,
        93.07798126001794,
        91.34190443642368,
        88.77992257550662,
        84.41889561314812,
        78.61877137510689,
        72.05176394644391,
        65.17405972205624,
        57.01763715032483,
        48.550235539853716,
        29.290056124944833,
        15.07947463146158,
        7.318051354238548,
        2.5622991788640492,
    ]
}
