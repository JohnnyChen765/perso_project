?	?e?Ŀ??@?e?Ŀ??@!?e?Ŀ??@	6?$	???6?$	???!6?$	???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?e?Ŀ??@ ?)U??r@1P)?E??@A??
???I?ɨ2?{@Y?F ^?/??*	??x?&?w@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2`r??ZC??!???V
W@){?\?&??1??????T@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle ??4????!.|?#??@)??4????1.|?#??@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip #?g]????!??N?|!@)??5[yɏ?1???y?r@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchC?O?}:??!8L4?H@)C?O?}:??18L4?H@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism|F"4????!P?wŃ@)????????1ُ8?߾	@:Preprocessing2F
Iterator::Model?????K??!'??Z@)?Y,E?e?1?p?zl???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 36.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no96?$	???I?j?sB@Q?P??O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ?)U??r@ ?)U??r@! ?)U??r@      ??!       "	P)?E??@P)?E??@!P)?E??@*      ??!       2	??
?????
???!??
???:	?ɨ2?{@?ɨ2?{@!?ɨ2?{@B      ??!       J	?F ^?/???F ^?/??!?F ^?/??R      ??!       Z	?F ^?/???F ^?/??!?F ^?/??b      ??!       JGPUY6?$	???b q?j?sB@y?P??O@?"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_6/Conv2DBackpropInputConv2DBackpropInput&??%????!&??%????0"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_9/Conv2DBackpropInputConv2DBackpropInputO+6ܲ??!:^?Rt??0"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_7/Conv2DBackpropInputConv2DBackpropInputW6Ȉ??!??MG[??0"]
4sequential_34/conv2d_transpose_64/conv2d_transpose_2Conv2DBackpropInput'/?Y???!}???K??"m
Dgradient_tape/sequential_33/conv2d_67/Conv2D_5/Conv2DBackpropInput_1Conv2DBackpropInput????'???!c?u?{y??"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_2/Conv2DBackpropInputConv2DBackpropInput?M?f??!j?{V?I??0"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_8/Conv2DBackpropInputConv2DBackpropInput:?	WH\??!q]a
???0"m
Dgradient_tape/sequential_33/conv2d_67/Conv2D_2/Conv2DBackpropInput_1Conv2DBackpropInput?B???ϓ?!?dR?O??"]
4sequential_34/conv2d_transpose_64/conv2d_transpose_1Conv2DBackpropInput?~?\??!%"hP???"k
@gradient_tape/sequential_33/conv2d_67/Conv2D/Conv2DBackpropInputConv2DBackpropInput??Hړ??!\Ck????0Q      Y@YIa???@a???"X@qD?4=?U@y	"???\S?"?

both?Your program is POTENTIALLY input-bound because 36.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?86.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 