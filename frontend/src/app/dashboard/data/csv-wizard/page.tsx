"use client";
import { ArrowLeft, UploadCloud, FileSpreadsheet, Settings2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export default function CSVWizardPage() {
  const [step, setStep] = useState(1);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/dashboard/data" className="btn-ghost">
          <ArrowLeft size={16} /> Back
        </Link>
        <h2 className="section-title">CSV Wizard</h2>
      </div>

      <div className="card space-y-8">
        <p className="text-gray-300">
          Upload any CSV file. Map the columns defining time, sensors, and labels to import your dataset effectively. 
        </p>
        
        {/* Wizard Steps */}
        <div className="flex justify-between relative mt-4">
          <div className="absolute top-1/2 left-0 w-full h-0.5 bg-gray-800 -z-10 -translate-y-1/2" />
          
          <div className={`flex flex-col items-center gap-2 \${step >= 1 ? "text-brand-400" : "text-gray-500"}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center \${step >= 1 ? "bg-brand-900 border border-brand-500" : "bg-gray-800"} font-bold`}>1</div>
            <span className="text-xs uppercase tracking-wider font-semibold">Upload</span>
          </div>
          
          <div className={`flex flex-col items-center gap-2 \${step >= 2 ? "text-brand-400" : "text-gray-500"}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center \${step >= 2 ? "bg-brand-900 border border-brand-500" : "bg-gray-800"} font-bold`}>2</div>
            <span className="text-xs uppercase tracking-wider font-semibold">Map Columns</span>
          </div>

          <div className={`flex flex-col items-center gap-2 \${step >= 3 ? "text-brand-400" : "text-gray-500"}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center \${step >= 3 ? "bg-brand-900 border border-brand-500" : "bg-gray-800"} font-bold`}>3</div>
            <span className="text-xs uppercase tracking-wider font-semibold">Review & Import</span>
          </div>
        </div>

        {/* Step 1 Content */}
        {step === 1 && (
          <div className="pt-8 space-y-4 text-center">
            <FileSpreadsheet size={48} className="mx-auto text-gray-600 mb-4" />
            <h3 className="text-lg font-bold text-white">Select a CSV file</h3>
            <p className="text-gray-400 text-sm max-w-md mx-auto">
              Please choose a raw CSV file to start. We support comma, semicolon, tab, and pipe separated files.
            </p>
            <div className="mt-6">
              <button 
                className="btn-primary" 
                onClick={() => setStep(2)}
              >
                Choose file
              </button>
            </div>
          </div>
        )}

        {/* Step 2 Placeholder */}
        {step === 2 && (
          <div className="pt-8 space-y-4 text-center">
            <Settings2 size={48} className="mx-auto text-brand-500 mb-4" />
            <h3 className="text-lg font-bold text-white">Map Your Data</h3>
            <p className="text-gray-400 text-sm max-w-md mx-auto">
              (Preview of your CSV goes here. You would select which column is timestamp, which are values, and optionally the label column.)
            </p>
            <div className="flex justify-center gap-4 mt-8">
              <button className="btn-ghost" onClick={() => setStep(1)}>Back</button>
              <button className="btn-primary" onClick={() => setStep(3)}>Next</button>
            </div>
          </div>
        )}

        {/* Step 3 Placeholder */}
        {step === 3 && (
          <div className="pt-8 space-y-4 text-center">
            <UploadCloud size={48} className="mx-auto text-green-500 mb-4" />
            <h3 className="text-lg font-bold text-white">Ready to Import</h3>
            <p className="text-gray-400 text-sm max-w-md mx-auto">
              We parsed 0 samples and are ready to import them into your project.
            </p>
            <div className="flex justify-center gap-4 mt-8">
              <button className="btn-ghost" onClick={() => setStep(2)}>Back</button>
              <button className="btn-primary" onClick={() => alert("Upload simulated!")}>Import Data</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
