/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2020 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

#ifndef CLOCK__H
#define CLOCK__H

class Clock
{
    public:
        Clock()
        {
            cudaEventCreate(&event_start);
            cudaEventCreate(&event_stop);
        }
        
        void start()
        {
            cudaEventRecord(event_start);
        }
        
        float stop()
        {
            cudaEventRecord(event_stop);
            cudaEventSynchronize(event_stop);
            float time;	
            cudaEventElapsedTime(&time, event_start, event_stop);
            return time/TO_SECONDS;
        }

    private:
        cudaEvent_t event_start, event_stop;
        static constexpr float TO_SECONDS = 1000;
};

#endif
