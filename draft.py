#+criterion(outputs2, labels2)
            #+ HSIC.hsic_normalized(outputs1, onehot_labels1, use_cuda=False) 
            # HSIC_loss = testHSIC.HSIC(outputs1, outputs2)
            # HSIC_loss1 = HSIC.hsic_normalized(outputs1, inputs2, use_cuda=False)-lambd*HSIC.hsic_normalized(outputs1, onehot_labels2, use_cuda=False) 
            # HSIC_loss2 = HSIC.hsic_normalized(outputs2, inputs1, use_cuda=False)-lambd*HSIC.hsic_normalized(outputs2, onehot_labels1, use_cuda=False) 
            # + HSIC.hsic_normalized(outputs1, onehot_labels1, sigma=None, use_cuda=False)

            # HSIC_loss = HSIC_loss1+HSIC_loss2
        


            # onehot_labels1 = torch.nn.functional.one_hot(labels1, num_classes=7)
            # onehot_labels2 = torch.nn.functional.one_hot(labels2, num_classes=7)



                # HSIC_loss1 = HSIC.hsic_normalized(outputs1, domains1, use_cuda=False) 
                # HSIC_loss2 = HSIC.hsic_normalized(outputs2, domains2, use_cuda=False) 
                # # print(HSIC_loss)
                # HSIC_loss = lambd*(HSIC_loss1+HSIC_loss2)

# ^(?=.*\bcartoon\b)(?=.*\bsketch\b).*$
#edit
