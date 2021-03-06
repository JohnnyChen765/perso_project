?	8L4H??@8L4H??@!8L4H??@	??????????????!???????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails68L4H??@g????@1W???5??A
??t??I?]/M @Y4d<J%<??*	<?O???S@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???=^H??!?hMoX?<@)vl?u???1Ϭ???q7@:Preprocessing2U
Iterator::Model::ParallelMapV2?P1?߄??!u?%??6@)?P1?߄??1u?%??6@:Preprocessing2F
Iterator::Model???p????!?;J?;?D@)?)9'?Ў?1Su??3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate4GV~???!Ϊ*???5@)??Z?a/??1,?<a^?(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice6Y???}?!o??h"@)6Y???}?1o??h"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"? ˂???!ĵe?M@)%?s}r?1? ??8J@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor1zn?+q?!R???2@)1zn?+q?1R???2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=ByG??!r_47??7@)?@??_?[?1?M?u@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 42.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?36.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???????Iȷ`???S@Qp3`??2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	g????@g????@!g????@      ??!       "	W???5??W???5??!W???5??*      ??!       2	
??t??
??t??!
??t??:	?]/M @?]/M @!?]/M @B      ??!       J	4d<J%<??4d<J%<??!4d<J%<??R      ??!       Z	4d<J%<??4d<J%<??!4d<J%<??b      ??!       JGPUY???????b qȷ`???S@yp3`??2@?"H
*gradient_tape/sequential_7/dense_22/MatMulMatMul??m?#ԇ?!??m?#ԇ?0":
sequential_7/dense_21/MatMulMatMul??:H?Ӈ?!5:???ӗ?0"H
*gradient_tape/sequential_7/dense_21/MatMulMatMul????????!??&*?i??0":
sequential_7/dense_22/MatMulMatMul????????!???k???0"H
,gradient_tape/sequential_7/dense_22/MatMul_1MatMulU*x ?)??!??]????"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch7B7T}?!??CEa???"X
7gradient_tape/sequential_7/dense_23/BiasAdd/BiasAddGradBiasAddGrad???ͩy?!A{s}?i??"H
,gradient_tape/sequential_7/dense_23/MatMul_1MatMul???ͩy?!?(EXj??":
sequential_7/dense_23/MatMulMatMul???ͩy?!;?3???0"H
*gradient_tape/sequential_7/dense_23/MatMulMatMul?x%d?u?!?-?t????0Q      Y@Y?? _??@af?*??W@q??ƌ?1@y??????"?
both?Your program is POTENTIALLY input-bound because 42.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?36.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?17.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 