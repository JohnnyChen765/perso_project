?	?B</]M@?B</]M@!?B</]M@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?B</]M@N&nĠK@1DP5z5@??A      ??IqqTn?@*	D?l???U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~??@??!?C?oA@)=?+J	???1?+XwN?=@:Preprocessing2U
Iterator::Model::ParallelMapV29?~߿y??!d?sA3@)9?~߿y??1d?sA3@:Preprocessing2F
Iterator::ModelH?`?????!}-?1?B@)W|C??u??1?Em? ]2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateP?Lۿ???!?-Uv?4@)???DR??1???t?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicekE???&|?!m?k?h@)kE???&|?1m?k?h@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??d#٫?!????O@)????g?r?1]d??[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?M?d?q?!jpя>@)?M?d?q?1jpя>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapc??????!=?2?.6@),am???R?1?dY|!??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??׬b?X@QS?	?T???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N&nĠK@N&nĠK@!N&nĠK@      ??!       "	DP5z5@??DP5z5@??!DP5z5@??*      ??!       2	      ??      ??!      ??:	qqTn?@qqTn?@!qqTn?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??׬b?X@yS?	?T????"I
+gradient_tape/sequential_17/dense_51/MatMulMatMul???L?-??!???L?-??0"I
+gradient_tape/sequential_17/dense_52/MatMulMatMul???L?-??!???L?-??0";
sequential_17/dense_52/MatMulMatMul?d??-??!???d!b??0";
sequential_17/dense_51/MatMulMatMulGAC?ۄ?!Q??)???0"I
-gradient_tape/sequential_17/dense_52/MatMul_1MatMul??-???!?????;??";
sequential_17/dense_53/MatMulMatMulp?Ll??{?!djs?ѵ??0"Y
8gradient_tape/sequential_17/dense_53/BiasAdd/BiasAddGradBiasAddGrad_?Yľ?{?!?O????"I
-gradient_tape/sequential_17/dense_53/MatMul_1MatMul_?Yľ?{?!?????Բ?"*

LogicalAnd
LogicalAnd>??G.w?!?n?C?G??"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch?d??-w?!??b3????Q      Y@Yu?E]t@a颋.??W@q???? U@yyp??'??"?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?84.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 