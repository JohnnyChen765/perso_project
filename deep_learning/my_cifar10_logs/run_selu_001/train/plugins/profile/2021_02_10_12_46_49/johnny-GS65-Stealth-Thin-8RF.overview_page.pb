?	r?_??a@r?_??a@!r?_??a@	P?6u????P?6u????!P?6u????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6r?_??a@??X3??`@1
e??k]@A??¼Ǚ??I???UG.@Y?ơ~???*	E?????V@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??]?????!۝S?D@)???8a ?1(8???A@:Preprocessing2U
Iterator::Model::ParallelMapV2F{????!B????@2@)F{????1B????@2@:Preprocessing2F
Iterator::ModelVIddY??!?^?'?~A@)?.\sG??1h?\?2?0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM?<i???!?6?44@)H??0~??1???3Ҹ(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?????P}?!???a^@)?????P}?1???a^@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?'G?`??!?P8l?@P@)??;Fzq?16V?ҵ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???u6?o?!?U?@)???u6?o?1?U?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9P?6u????Ih?vP?PX@Q?W???@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??X3??`@??X3??`@!??X3??`@      ??!       "	
e??k]@
e??k]@!
e??k]@*      ??!       2	??¼Ǚ????¼Ǚ??!??¼Ǚ??:	???UG.@???UG.@!???UG.@B      ??!       J	?ơ~????ơ~???!?ơ~???R      ??!       Z	?ơ~????ơ~???!?ơ~???b      ??!       JGPUYP?6u????b qh?vP?PX@y?W???@?":
sequential_1/dense_21/MatMulMatMul???	??!???	??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsKCa????!IJYѲ???"H
*gradient_tape/sequential_1/dense_21/MatMulMatMul?dQ??V??!p?mMi§?0"-
IteratorGetNext/_3_Send?]??u??!???Ƭ??"1
Nadam/Nadam/update/addAddV2??a?4w?![??K??"3
Nadam/Nadam/update/add_2AddV2??a?4w?!砥Ou???"3
Nadam/Nadam/update/add_1AddV2?ܭh?r?!?~??#??"9
Nadam/Nadam/update/truediv_3RealDiv?f-??q?!????=??"1
Nadam/Nadam/update/mul_1MulK[??x?n?!^v??5??"1
Nadam/Nadam/update/mul_3MulK[??x?n?!8??-??Q      Y@Y#%?T?/??ako??@?X@q?b}7??W@y|ft0???"?

both?Your program is POTENTIALLY input-bound because 94.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 