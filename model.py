import torch.nn as nn
import torch
import torchsnooper
import torch.nn.functional as F


class SiameseCNN(nn.Module):
    def __init__(self,args):
        super(SiameseCNN, self).__init__()

        self.embed=nn.Embedding(args.vocab_size,args.embedding_dim,padding_idx=args.pad_id)

        self.cnn1 = nn.Sequential(
                        nn.Conv1d(in_channels=args.embedding_dim,out_channels=100,kernel_size=2),
                        nn.ReLU())

        self.fc1 = nn.Linear(4*100+1, args.class_num)



    # @torchsnooper.snoop()
    def forward(self, input1, input2):
        output1 = self.embed(input1)
        output2 = self.embed(input2)

        output1=output1.permute(0,2,1)
        output2=output2.permute(0,2,1)


        output1=self.cnn1(output1)
        output2=self.cnn1(output2)

        output1=F.max_pool1d(output1,output1.size(2)).squeeze(2)
        output2=F.max_pool1d(output2,output2.size(2)).squeeze(2)

        s1=torch.cosine_similarity(output1, output2, dim=1)
        s1=s1.unsqueeze(1)
        s2=abs(output1-output2)#torch.dist(output1,output2,p=1)
        s3=output1*output2

        output=torch.cat((output1,output2,s1,s2,s3),1)
        output=self.fc1(output)

        return output



# if __name__ == '__main__':
    # x = torch.rand([16,20])
    # # print('x', x)
    # y = torch.rand([16,20])
    # # print('y', y)
    # embed_size=256
    # lr=0.01
    #
    # train_iter,val_iter,vocab_size,pad_id=load_data()
    # m=SiameseCNN(vocab_size,embed_size,pad_id)
    #
    # optimizer=torch.optim.Adam(m.parameters(),lr=lr)





    # for batch in train_iter:
    #     query1,query2,label=batch.query
