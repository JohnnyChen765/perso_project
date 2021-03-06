?	<???|O@<???|O@!<???|O@	2??IIӑ?2??IIӑ?!2??IIӑ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6<???|O@8L4H??M@1]????1??A?!???ɩ?I*p??a@Y?q??>s??*	??????[@2F
Iterator::Model~7ݲC???!H????H@)$??????1>'????@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX???<??!?K??7@)???Y???1@B]??Y3@:Preprocessing2U
Iterator::Model::ParallelMapV2??V|C???!x???1@)??V|C???1x???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(I?L?ٖ?!d#?)4@)鹅?D???1????Cr'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicef?(?7??!0p?? @)f?(?7??10p?? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?cϞˬ?!?Q&?K[I@)???RAEu?1Y?Ұ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?@?Ρu?!R/g=?@)?@?Ρu?1R/g=?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(~??k	??!??`? 6@)??ek}a?1???Cr???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no92??IIӑ?I?ڇ
*?X@Q???80.??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8L4H??M@8L4H??M@!8L4H??M@      ??!       "	]????1??]????1??!]????1??*      ??!       2	?!???ɩ??!???ɩ?!?!???ɩ?:	*p??a@*p??a@!*p??a@B      ??!       J	?q??>s???q??>s??!?q??>s??R      ??!       Z	?q??>s???q??>s??!?q??>s??b      ??!       JGPUY2??IIӑ?b q?ڇ
*?X@y???80.???"H
*gradient_tape/sequential_8/dense_24/MatMulMatMulM!!+???!M!!+???0":
sequential_8/dense_25/MatMulMatMulM!!+???!M!!+???0"H
*gradient_tape/sequential_8/dense_25/MatMulMatMulɣ?????!??h??=??0":
sequential_8/dense_24/MatMulMatMulɣ?????!??O?????0"H
,gradient_tape/sequential_8/dense_25/MatMul_1MatMul? ??d??!????????"H
,gradient_tape/sequential_8/dense_26/MatMul_1MatMul?Ĉ7??{?!???????"X
7gradient_tape/sequential_8/dense_25/BiasAdd/BiasAddGradBiasAddGrad?<]_!w?!??f???"*

LogicalAnd
LogicalAndɣ???v?!)?Pp0f??"$
Nadam/sub_3Subɣ???v?!f????ճ?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchɣ???v?!?|?p?E??Q      Y@Y?? _??@af?*??W@qƫ???RV@y?_??V???"?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?89.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 