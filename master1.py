from multiprocessing import Process

class ProcessParallel(object):
    """
    To Process the  functions parallely

    """    
    def __init__(self, *jobs):
        """
        """
        self.jobs = jobs
        self.processes = []

    def fork_processes(self):
        """
        Creates the process objects for given function deligates
        """
        for job in self.jobs:
            proc  = Process(target=job)
            self.processes.append(proc)

    def start_all(self):
        """
        Starts the functions process all together.
        """
        for proc in self.processes:
            proc.start()

    def join_all(self):
        """
        Waits untill all the functions executed.
        """
        for proc in self.processes:
            proc.join()


def two_sum(a=2, b=2):
    return a + b

def multiply(a=2, b=2):
    return a * b


#How to run:
if __name__ == '__main__':
    #note: two_sum, multiply can be replace with any python console scripts which
    #you wanted to run parallel..
    procs =  ProcessParallel(two_sum, multiply)
    #Add all the process in list
    procs.fork_processes()
    #starts  process execution 
    procs.start_all()
    #wait until all the process got executed
    procs.join_all()