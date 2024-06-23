
## 1- Install package
# qiskit-ibm-runtime==0.23.0

## 2- Upload token
# Save an IBM Quantum account.
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="")


from qiskit_ibm_runtime import QiskitRuntimeService, Batch , Sampler
import pickle


with open("qc_transpiled_decoupled_assigned_meas_{}_MFE".format('ibm_kyoto'), "rb") as fp:
   circuit_list_meas = pickle.load(fp)

service = QiskitRuntimeService(channel="ibm_quantum")

backend = service.backend("ibm_kyoto")
print(backend)


with Batch(service=service, backend=backend):
    sampler = Sampler()
    job = sampler.run(
        circuits=circuit_list_meas[630:760],  ## 3- Update here
        skip_transpilation=True,
        shots=10000,
    )
    print(job.job_id())
    result = job.result()

print('Simulation Finshed')


with open("results_qc_ibm_kyoto_MFE_630_760", "wb") as fp:  ## 4- Update here
    pickle.dump(result, fp)


print('Results dumped')
