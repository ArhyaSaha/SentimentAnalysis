import React from 'react';
import { Github, Coffee, Code2 } from 'lucide-react';

export const Footer = () => {
    return (
        <footer className="relative z-10 w-full mt-16">
            <div className="container mx-auto px-6 py-8">
                <div className="flex flex-col items-center gap-4 text-center">
                    <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                        <Code2 className="w-4 h-4" />
                        <span className="text-sm">Assignment by Arhya.</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                            <Coffee className="w-4 h-4" />
                            <span className="text-sm">Powered by Caffeine.</span>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    );
};
