?	M??StL$@M??StL$@!M??StL$@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-M??StL$@??2????1#?G?@A3??̝?I??֪M@*	?Zd;/Q@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?"[AӒ?!??qh??:@)<??kЗ??1
X?V?5@:Preprocessing2F
Iterator::Model???u???!M6>9;E@)????
??1?.I?'W5@:Preprocessing2U
Iterator::Model::ParallelMapV2p]1#???!?=ÜJ5@)p]1#???1?=ÜJ5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/?h?ґ?!?b$+`Q9@)<???	.??1,?Y?͂/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceE??@J?z?!? ?i?#@)E??@J?z?1? ?i?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!??????L@)ZF?=??n?1`7v???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoru?)?:l?!?"fp?@)u?)?:l?1?"fp?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?42.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI΢?N?N@Q2]s??C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??2??????2????!??2????      ??!       "	#?G?@#?G?@!#?G?@*      ??!       2	3??̝?3??̝?!3??̝?:	??֪M@??֪M@!??֪M@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q΢?N?N@y2]s??C@?"<
sequential_12/dense_227/MatMulMatMulz??n??!z??n??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits/?ω;??!I??$F??"J
,gradient_tape/sequential_12/dense_227/MatMulMatMul?4?u??!???????0"-
IteratorGetNext/_4_RecvtGV?????!??S#>E??"3
Nadam/Nadam/update/add_2AddV2??|??v?!,+?f???"1
Nadam/Nadam/update/addAddV2_Fe??u?!????f???"3
Nadam/Nadam/update/add_1AddV2?4?ur?!:????"9
Nadam/Nadam/update/truediv_3RealDiv ?=??_q?!?z??'??"<
sequential_12/dense_247/SoftmaxSoftmaxI:??Wq?!o???.=??"9
Nadam/Nadam/update/truediv_2RealDiv?Vk?Ip?!ٱk??A??Q      Y@Y#%?T?/??ako??@?X@q"C??1=@y?$?T:???"?
both?Your program is POTENTIALLY input-bound because 19.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?42.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?29.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 