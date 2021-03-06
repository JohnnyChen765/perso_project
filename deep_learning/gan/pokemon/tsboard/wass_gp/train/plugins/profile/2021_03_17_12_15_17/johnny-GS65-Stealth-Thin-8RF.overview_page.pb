?	??!?g{@??!?g{@!??!?g{@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??!?g{@?)?TP?@1/3l?uqz@A?V??y??I?9@0GO@*	?E????t@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2q?0'h???!????V@)??s?????16oE?b?T@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle u><K???!?3"?@)u><K???1?3"?@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip {-??1??!??q?"@)9(a????1U???@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch[??	m??!?F?z??@)[??	m??1?F?z??@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?D?$]3??!?i???@)??e1????1???DXP@:Preprocessing2F
Iterator::ModelJ$??(???!??'Hw0 @)???9]c?1p?|?-g??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?Q? ?@Qr????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)?TP?@?)?TP?@!?)?TP?@      ??!       "	/3l?uqz@/3l?uqz@!/3l?uqz@*      ??!       2	?V??y???V??y??!?V??y??:	?9@0GO@?9@0GO@!?9@0GO@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Q? ?@yr????X@?"m
Bgradient_tape/sequential_19/conv2d_39/Conv2D_5/Conv2DBackpropInputConv2DBackpropInput??ːr??!??ːr??0"m
Dgradient_tape/sequential_19/conv2d_39/Conv2D_8/Conv2DBackpropInput_1Conv2DBackpropInput?Q??G??!V?D_?ܥ?"m
Bgradient_tape/sequential_19/conv2d_39/Conv2D_2/Conv2DBackpropInputConv2DBackpropInput???Ζ6??!?9Rc<??0"m
Bgradient_tape/sequential_19/conv2d_39/Conv2D_7/Conv2DBackpropInputConv2DBackpropInput?T&?ߔ?!y???s??0"[
2sequential_20/conv2d_transpose_36/conv2d_transposeConv2DBackpropInputTBȒd??!?????"k
@gradient_tape/sequential_19/conv2d_39/Conv2D/Conv2DBackpropInputConv2DBackpropInputR???DP??!*0??/???0"]
4sequential_20/conv2d_transpose_36/conv2d_transpose_1Conv2DBackpropInput?????!?o?ZG??"m
Bgradient_tape/sequential_19/conv2d_39/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput$t?-y??!|>?{????0"m
Bgradient_tape/sequential_19/conv2d_39/Conv2D_9/Conv2DBackpropInputConv2DBackpropInput?n???!???????0"m
Bgradient_tape/sequential_19/conv2d_39/Conv2D_8/Conv2DBackpropInputConv2DBackpropInput?ſ? ??!??&?y??0Q      Y@YIa???@a???"X@q??iN?!>@y(?o???G?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 