

std::vector<std::vector<float>> conv_is = {
{0.93942745,0.55185495,0.32203628,0.94133963,0.9837407 ,0.75818242, 0.59415383,0.61135525,0.14140573,0.08703763,0.68637563,0.47665525, 0.79756787,0.68911478,0.66759872,0.07458822,0.03530135,0.2321531 , 0.32024295,0.52081356,0.13374231,0.85849583,0.95442983,0.55410856, 0.68339854,0.45956486,0.78113762,0.5022657 ,0.91228196,0.36593022, 0.76411951,0.57131084,0.80503139,0.84267779,0.08511586,0.00774067, 0.77879095,0.07757862,0.07472336,0.56295807,0.80276164,0.95717371, 0.39242439,0.7366593 ,0.85942225,0.82878644,0.89919649,0.1871433 , 0.66887897,0.15748321,0.57362008,0.20678843,0.20172773,0.62400081, 0.40874996,0.18440421,0.68762811,0.43844716,0.21890678,0.16600624, 0.82697926,0.26075221,0.41527578,0.6234682 }};
std::vector<std::vector<float>> conv_k1s = {
{2.41774744e-01,7.43976695e-01,6.48184510e-01,9.54601301e-02, 1.32006138e-01,5.87572218e-01,6.68828151e-01,7.81549267e-01, 2.95658359e-01,7.17031886e-01,4.50584176e-01,8.32662581e-01, 6.83498381e-01,1.72706202e-01,8.09901781e-04,9.90164312e-01, 2.54599048e-01,4.56788559e-02,8.57770668e-01,2.79880036e-01, 9.14883672e-01,4.80987222e-01,5.91284995e-01,8.53564425e-02, 3.86506686e-01,3.80461225e-01,6.44755446e-01,9.68261581e-01, 4.40077081e-01,5.85645179e-01,1.15850711e-01,7.16950657e-01, 6.70804580e-01,9.03618636e-01,7.85679060e-01,2.98518461e-01, 7.62380625e-01,2.47117044e-01,6.66582299e-01,2.85421307e-01, 8.57891717e-01,6.45865132e-02,1.83816726e-01,8.20989027e-01, 4.28499356e-02,5.80749404e-01,9.78440513e-01,7.16122883e-01, 2.24321263e-01,7.76272494e-01,8.20487560e-01,4.48120627e-01, 5.94698466e-01,9.08544123e-01,6.78392612e-01,3.42330033e-01, 7.15811907e-01,1.21346056e-01,5.19667815e-02,1.23777655e-02, 3.75125663e-01,5.42686185e-02,9.23216489e-01,8.05331369e-01, 5.64446420e-01,7.30630274e-03,3.73750384e-01,5.06715941e-01, 8.56349509e-01,9.75103759e-01,6.17595168e-01,4.88253912e-02}};
std::vector<std::vector<float>> conv_k2s = {
{0.46020922,0.22288028,0.42868289,0.54124292,0.38237708,0.32606336, 0.88419322,0.11125517,0.33563169,0.54409747,0.49556901,0.38408629, 0.61045428,0.85477622,0.12300465,0.73125523}};
std::vector<std::vector<float>> conv_os = {
{0.7036658 ,0.34078678,0.57615536,0.7274375 ,0.6042534 ,0.5152634 , 1.367573  ,0.17207728,0.44845244,0.72699285,0.8236661 ,0.638375  , 1.221648  ,1.7105879 ,0.2632985 ,1.5652937 ,0.89582795,0.43385133, 0.6822394 ,0.8613762 ,0.91580963,0.7809358 ,1.447151  ,0.1820903 , 0.8190668 ,1.3278013 ,1.0239437 ,0.79359823,1.608389  ,2.252114  , 0.22054432,1.3111227 ,1.1026729 ,0.53402674,0.8640669 ,1.0909464 , 1.0630736 ,0.9065119 ,2.4485862 ,0.30809766,0.9350358 ,1.5158004 , 1.118768  ,0.86709094,1.6824809 ,2.3558598 ,0.36933678,2.195685  , 1.2513288 ,0.6060211 ,1.2061806 ,1.5228895 ,1.3457876 ,1.1475898 , 2.7026358 ,0.3400639 ,1.1521263 ,1.867729  ,1.626303  ,1.2604514 , 2.0430734 ,2.8607721 ,0.38934287,2.3146198 ,1.4449552 ,0.69979477, 1.0660866 ,1.3460108 ,1.2153316 ,1.0363463 ,2.6276035 ,0.33062285, 1.1127968 ,1.8039713 ,1.5454823 ,1.1978121 ,2.251109  ,3.1520696 , 0.4428939 ,2.6329775 ,1.1419293 ,0.5530387 ,1.1465417 ,1.4475912 , 1.2340006 ,1.0522659 ,2.071913  ,0.2607021 ,1.0820829 ,1.7541804 , 1.4005345 ,1.0854716 ,1.9977978 ,2.797376  ,0.37647393,2.2381148 , 0.73723346,0.35704368,0.44361535,0.5600962 ,0.7112997 ,0.6065447 , 1.1956251 ,0.15044163,0.72499627,1.1753021 ,1.0405997 ,0.80650735, 1.4302851 ,2.0027277 ,0.14986604,0.89094454,0.93994176,0.45521575, 0.61954355,0.78221816,0.7177043 ,0.612006  ,1.7076557 ,0.21486877, 0.76583827,1.2415117 ,0.92482775,0.71677935,1.4856492 ,2.0802503 , 0.27430877,1.630749  ,0.9882991 ,0.47863528,0.92764413,1.1712172 , 1.1440864 ,0.9755937 ,1.8386965 ,0.23135722,1.0485092 ,1.6997536 , 1.4320372 ,1.1098875 ,1.8333204 ,2.5670695 ,0.27884495,1.6577164 , 1.1626023 ,0.5630507 ,0.78693277,0.99355906,0.93850404,0.800288  , 2.275286  ,0.28629184,0.8594675 ,1.3932954 ,1.2134312 ,0.94045883, 1.9078816 ,2.6714725 ,0.390162  ,2.3194895 ,1.3395783 ,0.64876056, 1.1850641 ,1.4962283 ,1.2960356 ,1.1051649 ,2.4816365 ,0.31225628, 1.0993545 ,1.7821797 ,1.6588237 ,1.2856563 ,2.3679368 ,3.3156555 , 0.37287295,2.2167072 ,1.2802669 ,0.6200359 ,1.11483   ,1.4075528 , 1.0592571 ,0.9032575 ,2.3665917 ,0.29778057,0.91902566,1.489846  , 1.2624749 ,0.9784697 ,1.8862422 ,2.6411724 ,0.40090075,2.3833308 , 1.0276849 ,0.49770993,0.7119922 ,0.8989412 ,0.796715  ,0.6793806 , 1.8928974 ,0.23817714,0.67416805,1.0929037 ,1.012172  ,0.78447473, 1.6724179 ,2.3417692 ,0.2981257 ,1.7723393 ,0.7686606 ,0.3722639 , 0.85899484,1.0845426 ,0.9339909 ,0.7964395 ,1.6867718 ,0.21224101, 0.6997929 ,1.1344446 ,0.9781845 ,0.75813305,1.3331568 ,1.8667259 , 0.22294691,1.325406  ,0.9899819 ,0.4794503 ,0.64322805,0.8121215 , 0.7587257 ,0.64698607,2.2898684 ,0.2881267 ,0.6735514 ,1.091904  , 1.1353439 ,0.87993807,1.317034  ,1.8441502 ,0.2715314 ,1.6142377 , 0.96968263,0.46961927,0.71279097,0.89994967,0.8863191 ,0.75578845, 1.3376572 ,0.16831307,0.7521477 ,1.2193177 ,0.7924579 ,0.6141873 , 1.6192632 ,2.2673404 ,0.26186514,1.5567725 ,0.9458059 ,0.45805576, 0.90035516,1.136763  ,1.1392176 ,0.9714419 ,2.6547413 ,0.33403748, 0.92200017,1.4946681 ,1.2582604 ,0.97520334,1.4637712 ,2.049616  , 0.3289737 ,1.9557289 ,0.9957112 ,0.48222497,0.98098296,1.2385614 , 1.0251755 ,0.874195  ,2.1257446 ,0.26747555,0.8716174 ,1.4129918 , 1.1369271 ,0.881165  ,1.5155293 ,2.1220891 ,0.32806516,1.9503275 , 1.1671629 ,0.5652594 ,1.1527518 ,1.4554318 ,1.2893419 ,1.099457  , 2.0671282 ,0.26010004,1.1048502 ,1.7910889 ,1.362671  ,1.0561259 , 2.0317183 ,2.8448722 ,0.36892548,2.1932397 ,0.8149367 ,0.39467552, 0.9083776 ,1.1468918 ,1.0604718 ,0.9042932 ,2.2013123 ,0.276984  , 0.9523308 ,1.5438374 ,1.3751086 ,1.0657655 ,1.3634553 ,1.9091506 , 0.30617863,1.8202134 ,1.0935812 ,0.5296236 ,0.6314226 ,0.7972163 , 0.9179915 ,0.7827964 ,1.8503278 ,0.23282075,0.8738019 ,1.4165331 , 1.349482  ,1.0459039 ,2.0244997 ,2.8347645 ,0.31688344,1.883853  , 1.069768  ,0.51809084,1.0453008 ,1.3197674 ,1.0096576 ,0.86096257, 1.74572   ,0.2196583 ,1.0388832 ,1.6841488 ,1.1867894 ,0.91981035, 1.8032818 ,2.5250087 ,0.31922707,1.8977857 ,1.0281276 ,0.4979243 , 0.9094582 ,1.1482562 ,1.1749513 ,1.0019131 ,2.0294394 ,0.25535777, 1.0292875 ,1.6685929 ,1.5798907 ,1.2244799 ,1.941752  ,2.7188988 , 0.3419668 ,2.032972  ,0.9011426 ,0.43642524,0.7431134 ,0.938234  , 0.79551077,0.6783537 ,1.651712  ,0.20782955,0.738035  ,1.1964394 , 0.9895917 ,0.7669741 ,1.6486924 ,2.3085482 ,0.31839302,1.8928274 , 1.0773536 ,0.5217645 ,1.2625817 ,1.5940999 ,1.180646  ,1.006769  , 2.4579515 ,0.30927607,0.9424567 ,1.5278305 ,1.372552  ,1.063784  , 1.6224995 ,2.271872  ,0.36709505,2.182358  ,0.94743   ,0.4588423 , 0.7064831 ,0.8919856 ,0.722075  ,0.615733  ,1.7548277 ,0.22080429, 0.47471327,0.7695647 ,0.9786303 ,0.7584786 ,1.480733  ,2.0733664 , 0.3192323 ,1.8978169 ,0.9180385 ,0.44460794,0.62020534,0.7830537 , 0.86923665,0.7412218 ,1.6798079 ,0.21136478,0.7020138 ,1.138045  , 1.0238432 ,0.7935204 ,1.5868504 ,2.221955  ,0.19932544,1.1849778 , 1.0893693 ,0.5275838 ,0.8006388 ,1.0108639 ,0.817669  ,0.69724864, 2.4160872 ,0.30400842,0.78947943,1.2798368 ,1.0450412 ,0.8099497 , 1.3629718 ,1.9084736 ,0.29699177,1.7655982 ,1.202264  ,0.58225894, 0.86066824,1.0866554 ,0.88907605,0.7581394 ,2.0185072 ,0.25398225, 0.66349864,1.0756074 ,1.1660653 ,0.9037484 ,1.869414  ,2.617609  , 0.34195924,2.032927  ,0.8306289 ,0.40227526,0.77689856,0.98089015, 0.9346196 ,0.7969756 ,1.617954  ,0.20358191,0.7776498 ,1.2606596 , 0.94874537,0.7353165 ,1.3295069 ,1.8616151 ,0.21148627,1.2572731 , 0.9318876 ,0.4513151 ,0.99622077,1.2578002 ,1.0008576 ,0.8534585 , 2.2755191 ,0.2863212 ,0.7783579 ,1.2618074 ,1.2148898 ,0.94158936, 1.5382822 ,2.1539483 ,0.386618  ,2.2984204 ,0.69344544,0.33583704, 0.8517424 ,1.0753858 ,0.93778825,0.7996776 ,1.2388452 ,0.15587987, 0.6944337 ,1.1257566 ,0.9468743 ,0.73386633,1.392495  ,1.9498129 , 0.24093291,1.4323317 ,0.7822127 ,0.3788272 ,0.63173485,0.7976106 , 0.9139552 ,0.7793545 ,1.9393367 ,0.24402045,0.8930998 ,1.4478172 , 1.1346809 ,0.87942415,1.2119969 ,1.6970742 ,0.24271487,1.4429252 , 0.9987278 ,0.4836859 ,0.74356437,0.9388033 ,0.8637144 ,0.7365128 , 1.7749188 ,0.22333227,0.7876184 ,1.2768198 ,0.97290623,0.75404215, 1.7028948 ,2.3844438 ,0.35495394,2.1101797 ,0.98279214,0.47596824, 1.1224529 ,1.4171772 ,1.2406445 ,1.0579313 ,2.055951  ,0.25869367, 1.0169103 ,1.6485282 ,1.3681519 ,1.0603738 ,1.8341916 ,2.5682895 , 0.33185214,1.972841  ,0.87215066,0.42238435,0.9199249 ,1.1614712 , 0.8904847 ,0.7593405 ,1.9414587 ,0.24428745,0.78340065,1.2699823 , 1.0769778 ,0.83470196,1.3040626 ,1.8259871 ,0.33264175,1.977535  }};
std::vector<std::string> conv_modules = {
R"#(HloModule cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_0__XlaNumResourceArgs_0_.18

ENTRY %cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_0__XlaNumResourceArgs_0_.18 (arg0.1: f32[1,8,8,1], arg1.2: f32[3,3,1,8], arg2.3: f32[1,1,1,16]) -> f32[1,6,6,16] {
  %constant.12 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu_1"}
  %broadcast.13 = f32[1,6,6,16]{3,2,1,0} broadcast(f32[] %constant.12), dimensions={}, metadata={op_type="Relu" op_name="Relu_1"}
  %constant.8 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9 = f32[1,6,6,8]{3,2,1,0} broadcast(f32[] %constant.8), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %arg0.1 = f32[1,8,8,1]{3,2,1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.4 = f32[1,8,8,1]{3,2,1,0} reshape(f32[1,8,8,1]{3,2,1,0} %arg0.1)
  %arg1.2 = f32[3,3,1,8]{3,2,1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.5 = f32[3,3,1,8]{3,2,1,0} reshape(f32[3,3,1,8]{3,2,1,0} %arg1.2)
  %convolution.7 = f32[1,6,6,8]{3,2,1,0} convolution(f32[1,8,8,1]{3,2,1,0} %reshape.4, f32[3,3,1,8]{3,2,1,0} %reshape.5), window={size=3x3}, dim_labels=b01f_01io->b01f, metadata={op_type="Conv2D" op_name="Conv2D"}
  %maximum.10 = f32[1,6,6,8]{3,2,1,0} maximum(f32[1,6,6,8]{3,2,1,0} %broadcast.9, f32[1,6,6,8]{3,2,1,0} %convolution.7), metadata={op_type="Relu" op_name="Relu"}
  %arg2.3 = f32[1,1,1,16]{3,2,1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.6 = f32[1,1,1,16]{3,2,1,0} reshape(f32[1,1,1,16]{3,2,1,0} %arg2.3)
  %convolution.11 = f32[1,6,6,16]{3,2,1,0} convolution(f32[1,6,6,8]{3,2,1,0} %maximum.10, f32[1,1,1,16]{3,2,1,0} %reshape.6), window={size=1x1}, dim_labels=b01f_01io->b01f, feature_group_count=8, metadata={op_type="Conv2D" op_name="Conv2D_1"}
  %maximum.14 = f32[1,6,6,16]{3,2,1,0} maximum(f32[1,6,6,16]{3,2,1,0} %broadcast.13, f32[1,6,6,16]{3,2,1,0} %convolution.11), metadata={op_type="Relu" op_name="Relu_1"}
  %reshape.15 = f32[1,6,6,16]{3,2,1,0} reshape(f32[1,6,6,16]{3,2,1,0} %maximum.14), metadata={op_name="XLA_Retvals"}
  %tuple.16 = (f32[1,6,6,16]{3,2,1,0}) tuple(f32[1,6,6,16]{3,2,1,0} %reshape.15), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.17 = f32[1,6,6,16]{3,2,1,0} get-tuple-element((f32[1,6,6,16]{3,2,1,0}) %tuple.16), index=0, metadata={op_name="XLA_Retvals"}
}

)#"};