?	?q???O@?q???O@!?q???O@	d??#????d??#????!d??#????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?q???O@`??fM@1;??u??Af?B,c??I???Z?@Y!=E7??*	;?O???T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??)r???!??$?<@)??V????1;|C/r58@:Preprocessing2U
Iterator::Model::ParallelMapV2???f???!??1??7@)???f???1??1??7@:Preprocessing2F
Iterator::Model?:pΈ??!_^r??E@)[^??6S??19?q?D4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?+I?????!?Aw?c?#@)?+I?????1?Aw?c?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???QI???!䶯??o3@)c'??>??1,??? #@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???o?4??!????PQL@)??ŉ?vt?1?!!r?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??9?n?!MK???@)??9?n?1MK???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap>x?҆Ò?!t?Χ`?5@)XuV?1a?1?$??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9e??#????I?=h?ՖX@Q,????^??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	`??fM@`??fM@!`??fM@      ??!       "	;??u??;??u??!;??u??*      ??!       2	f?B,c??f?B,c??!f?B,c??:	???Z?@???Z?@!???Z?@B      ??!       J	!=E7??!=E7??!!=E7??R      ??!       Z	!=E7??!=E7??!!=E7??b      ??!       JGPUYe??#????b q?=h?ՖX@y,????^???"I
+gradient_tape/sequential_14/dense_43/MatMulMatMulf??D???!f??D???0";
sequential_14/dense_42/MatMulMatMul?t????!??f\,??0"I
+gradient_tape/sequential_14/dense_42/MatMulMatMulF?'???!?n??????0";
sequential_14/dense_43/MatMulMatMulF?'???!?i?A?x??0"I
-gradient_tape/sequential_14/dense_43/MatMul_1MatMul??V?^???!?
??pg??";
sequential_14/dense_44/MatMulMatMul?wkb?Oz?!?y??a???0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch???mNz?!?n??????"I
-gradient_tape/sequential_14/dense_44/MatMul_1MatMul???mNz?!b???~???"$
Nadam/sub_3SubF?'?u?!6??>??"Y
8gradient_tape/sequential_14/dense_44/BiasAdd/BiasAddGradBiasAddGradF?'?u?!
?dY?_??Q      Y@Yu?E]t@a颋.??W@q?????S@y??ݰg??"?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?78.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 