?	??^D??`@??^D??`@!??^D??`@	??F???????F?????!??F?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??^D??`@f?B,?_@1*Wx???@A???;jL??I?I?U@Y???b???*	????S@2U
Iterator::Model::ParallelMapV2??N??:??!?????=@)??N??:??1?????=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat+~??7??!?~??<2;@)????y??1??O)?5@:Preprocessing2F
Iterator::Model???G?)??!>???0?H@)?c?1??1?????Y3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?d??~???!;??Y??2@)??H¾}?1Je?bI#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceG=D?;?}?!,GPW?"@)G=D?;?}?1,GPW?"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5??o?hp?!??EO@)5??o?hp?1??EO@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???:8أ?!?C!?oI@)???8m?1;?HD?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??F?????I[w??]ZX@Q}????Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	f?B,?_@f?B,?_@!f?B,?_@      ??!       "	*Wx???@*Wx???@!*Wx???@*      ??!       2	???;jL?????;jL??!???;jL??:	?I?U@?I?U@!?I?U@B      ??!       J	???b??????b???!???b???R      ??!       Z	???b??????b???!???b???b      ??!       JGPUY??F?????b q[w??]ZX@y}????Q@?"<
sequential_16/dense_311/MatMulMatMul???c??!???c??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits?rɣ???!L??ƪ???"-
IteratorGetNext/_3_Send????J'??!???h}v??"J
,gradient_tape/sequential_16/dense_311/MatMulMatMul??r?-??!`?????0"1
Nadam/Nadam/update/addAddV2?#??W?y?!?\?s???"3
Nadam/Nadam/update/add_2AddV2??cvWLx?!GJ?`???"3
Nadam/Nadam/update/add_1AddV2k??U9r?!^u!???"9
Nadam/Nadam/update/truediv_3RealDiv[??H9r?!4??%????"9
Nadam/Nadam/update/truediv_2RealDiv?_c?q?!/?r??O??"1
Nadam/Nadam/update/SqrtSqrtld ??o?!?v?zL??Q      Y@Y#%?T?/??ako??@?X@qP?N?V@y??e?g???"?
both?Your program is POTENTIALLY input-bound because 93.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?88.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 