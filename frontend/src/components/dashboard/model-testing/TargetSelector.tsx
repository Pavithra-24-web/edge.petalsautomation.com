"use client";

import { useState } from "react";
import { Monitor, ChevronDown } from "lucide-react";

interface TargetSelectorProps {
  devices: string[];
  selected: string;
}

export default function TargetSelector({ devices, selected }: TargetSelectorProps) {
  const [open, setOpen] = useState(false);
  const [activeDevice, setActiveDevice] = useState(selected);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 bg-gray-900 border border-gray-700 text-white
                   text-xs font-medium px-3 py-1.5 rounded-lg hover:bg-gray-800
                   transition-colors shadow-sm"
      >
        <Monitor size={13} className="text-gray-400" />
        <span className="truncate max-w-[160px]">Target: {activeDevice}</span>
        <ChevronDown size={13} className="text-gray-400 flex-shrink-0" />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 bg-slate-900 border border-slate-700
                        rounded-lg shadow-lg min-w-[200px] py-1">
          {devices.map((d) => (
            <button
              key={d}
              onClick={() => { setActiveDevice(d); setOpen(false); }}
              className={`w-full text-left px-4 py-2 text-sm transition-colors
                ${d === activeDevice
                  ? "text-blue-300 bg-blue-950/40 font-medium"
                  : "text-slate-200 hover:bg-slate-800"}`}
            >
              {d}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
