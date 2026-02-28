Safe-Wellness Engine: Hybrid Edge-SLM Prototype
This repository contains the functional prototype for the Safe-Wellness Engine, a multi-layer framework designed for next-generation healthcare systems. The system integrates real-time deterministic guardrails with a Small Language Model (SLM) to provide safe, privacy-preserving wellness coaching.

1. Project Concept
The prototype addresses the critical challenge of deploying Large Language Models (LLMs) in health contexts: balancing conversational intelligence with medical safety and edge resource constraints. It operates on a Bimodal Execution Path:

    a.The Critical Path (Layer 2): A high-speed deterministic layer that monitors vital signs (HR, SpO_2, Stress). If life-threatening anomalies are detected, it bypasses the AI entirely to deliver instantaneous emergency alerts.
    
    b.The Cognitive Path (Layer 3 & 4): A generative layer powered by a quantized SLM that provides personalized wellness tips only when user vitals are within safe thresholds.

2. Core Algorithms & LogicThe prototype implements a Tri-Stage Guardrail Architecture to ensure ethical and technical robustness:

    Phase 1: Deterministic Filtering (Layer 2)The Python script evaluates incoming sensor data against clinical thresholds.Logic: Monitors if HR > 150 BPM, SpO_2 < 90%, or Stress is "Critical".Performance: This phase typically executes in under 0.03 ms, ensuring "near real-time" safety interrupts.

    Phase 2: Constrained Inference (Layer 3)If the user is safe, the system invokes the Phi-3-mini-4k-instruct model (3.8B parameters).Alignment: A rigid System Prompt Framework anchors the SLM’s persona as a "Wellness Coach".Policy: Explicitly forbids medical diagnosis, pharmaceutical prescriptions, and the use of clinical jargon.

    Phase 3: Post-Verification (Layer 4)The script performs a final safety scan on generated text.Verification: A high-sensitivity keyword-regex filter checks for restricted terms like "cancer," "pill," or "diagnose".Escalation: If a trigger is found, the system replaces the output with a pre-validated ESCALATION disclaimer.

3. Installation & Execution
    Prerequisites
    Ensure you have Python 3.11 and CUDA drivers installed for GPU acceleration.

        Step 1: Set up the Virtual Environment
        
                # Navigate to the project directory
                cd "~/Simulation_Prototype"

                # Create and activate the virtual environment
                python3.11 -m venv .venv311
                source .venv311/bin/activate
        
        Step 2: Install Required Libraries

                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                pip install transformers accelerate psutil pandas 

        Step 3: Run the Prototype
                
                # Run the validation script
                python SLM_hybrid_performance_v3.py
4. Data Output & Purpose
The simulation generates a detailed performance log: validation_100_cases_results.csv.

Purpose of the Output File
The CSV serves as empirical evidence for the system's "real-time" and "efficiency" claims. It allows for the analysis of:

    a.Latency Distribution: Verifying the sub-1ms speed of the Critical Path vs. the ~25s processing time of the Cognitive Path.

    b.Safety Alignment: Auditing the Output column to ensure no medical hallucinations bypassed the Layer 4 filters.

    c.System Stability: Tracking CPU_Temp and Power to evaluate hardware load during edge-based SLM inference.

    d.CSV Column Definitions

            Column      Description
            Iteration	The specific test case ID (1-100).
            Mode	    Resultant state: EMERGENCY, WELLNESS, or ESCALATION.
            Latency	    End-to-end processing time in milliseconds.
            Power	    Simulated power metric based on CPU load.
            CPU_Temp	Real-time hardware temperature monitoring.
            Output	    The final text delivered to the user or safety disclaimer.