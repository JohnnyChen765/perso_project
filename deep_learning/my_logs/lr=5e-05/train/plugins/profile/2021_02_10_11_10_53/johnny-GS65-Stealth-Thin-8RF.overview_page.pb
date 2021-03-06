?	w??3a@w??3a@!w??3a@	?}ж{????}ж{???!?}ж{???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6w??3a@?z?|"`@1?J?4?@A:vP????I??>???@Y????C??*	??(\??U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?8'0???!??cnC9@)?w~Q????1?W5/85@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap
H?`???!??{{?=@)???zݒ?1??H5@:Preprocessing2F
Iterator::Model?Z??Ρ?!?h ,??C@)??u?ݑ?1????3@:Preprocessing2U
Iterator::Model::ParallelMapV24??𽿑?!?1U?3@)4??𽿑?1?1U?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?????!v?Eet!@)?????1v?Eet!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV?P?????!
???N@)~?[?~lr?1g}$L?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorh??n?l?!C0??,@)h??n?l?1C0??,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?}ж{???I??[?YFX@Qު?E-?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?z?|"`@?z?|"`@!?z?|"`@      ??!       "	?J?4?@?J?4?@!?J?4?@*      ??!       2	:vP????:vP????!:vP????:	??>???@??>???@!??>???@B      ??!       J	????C??????C??!????C??R      ??!       Z	????C??????C??!????C??b      ??!       JGPUY?}ж{???b q??[?YFX@yު?E-?@?"<
sequential_17/dense_332/MatMulMatMul\Ϻ????!\Ϻ????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits3??gQ;??!????8ޢ?"-
IteratorGetNext/_3_Send1p:WC??!?b????"J
,gradient_tape/sequential_17/dense_332/MatMulMatMulfR???!?8$?s??0"3
Nadam/Nadam/update/add_2AddV2?]W?bw?!=?
/?_??"1
Nadam/Nadam/update/addAddV2`~Ұ?Rv?!????"??"3
Nadam/Nadam/update_9/mul_4Mul?#?<r?!????F6??"3
Nadam/Nadam/update_9/SqrtSqrt?<?r?!???fW??"3
Nadam/Nadam/update/add_1AddV2]j?h?q?!v?{??g??"1
Nadam/Nadam/update/mul_1Mul????o?!????f??Q      Y@Y#%?T?/??ako??@?X@q`/?v ?W@y?A=Jϛ??"?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?94.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 